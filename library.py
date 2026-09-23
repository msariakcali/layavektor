"""Kitap kütüphanesi: PDF -> paragraflar, kitap bilgisi (başlık, dil, bölümler), tekrar eden baskıların ayıklanması.
book_server.py ve match_server.py ortak kullanır.

  BOOKS_DIR=<klasör>[;<klasör>...]   alt klasörler dahil tüm PDF'ler
                                     (varsayılan: bu klasörün bir üstündeki PDF'ler + onun altındaki books/ klasörü)
Her PDF ayrı önbelleğe alınır (data/books/pdf/<özet>.json, git dışı): yeni kitap eklenince sadece o işlenir.
Kitap bilgisi önceliği: books.json'daki elle girilmiş başlık/bölümler > PDF'in içindekiler kaydı > kapaktaki en büyük
puntolu yazı ve sabit sayfa aralıkları.
Tekrarlar: bir kitabın 8 kelimelik dizilerinin %25'ten fazlası daha büyük bir kitapta da geçiyorsa (aynı kitabın başka
baskısı, ciltlerin birleşik hali) küçük olan atlanır. Çeviriler farklı dilde olduğu için tekrar sayılmaz.
"""
import glob
import hashlib
import json
import os
import re
import unicodedata
import zlib
from collections import Counter

import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
DEFAULT_PATTERNS = [os.path.join(PARENT, "*.pdf"), os.path.join(PARENT, "books", "**", "*.pdf")]
CHUNK_WORDS, STRIDE = 120, 90          # ~250 token: v3'ün 256 token sınırına sığar
WINDOW_PAGES = 20                      # bölüm bilgisi olmayan kitaplarda bölüm yerine sayfa aralığı
MIN_CHUNKS = 5                         # daha az paragraf çıkan PDF (ör. taranmış görüntü) atlanır
DUP_SHINGLE, DUP_CONTAIN = 8, 0.25        # gözden geçirilmiş baskılar %26–47, ilgisiz kitaplar en fazla %13 örtüşüyor
CACHE = os.path.join(HERE, "data", "books")
VERSION = 2                            # önbellek biçimi değişince artır

_TR = str.maketrans({"İ": "i", "I": "ı"})


def norm(s):
    return s.translate(_TR).lower()


# ------------------------------------------------------------------ tek PDF
def pdf_chunks(doc):
    pages = [p.get_text() for p in doc]
    # sayfa başlığı/altlığı: birçok sayfada tekrar eden satırlar
    freq = Counter(norm(l.strip()) for t in pages for l in set(t.split("\n")) if l.strip())
    headers = {l for l, c in freq.items() if c >= max(5, len(pages) // 10)}
    words = []   # (kelime, sayfa)
    for pno, t in enumerate(pages, 1):
        if len(re.findall(r"(?:\.\s*){5,}", t)) >= 5:   # içindekiler sayfası: başlıklar arama sonucu olarak işe yaramaz
            continue
        lines = []
        for l in t.split("\n"):
            s = l.strip()
            if (not s or s in "•·-–" or re.fullmatch(r"[\divxlcIVXLC]{1,4}", s) or re.search(r"(\.\s*){5,}", s)
                    or norm(s) in headers):
                continue
            lines.append(s)
        text = re.sub(r"­\s*", "", " ".join(lines))          # yumuşak tire: "yürütme­ lere"
        text = re.sub(r"(\w)- (\w)", r"\1\2", text)               # satır sonunda bölünmüş kelimeler
        words += [(w, pno) for w in text.split()]
    chunks = []
    for i in range(0, max(1, len(words) - CHUNK_WORDS // 3), STRIDE):
        part = words[i:i + CHUNK_WORDS]
        text = " ".join(w for w, _ in part)
        if not part or sum(ch.isalpha() for ch in text) < 0.6 * len(text):   # boş sayfa / içindekiler / tablo artığı
            continue
        chunks.append({"page": part[0][1], "page_end": part[-1][1], "text": text})
    return chunks


def cover_title(doc, max_pages=4):
    """İlk sayfalarda en büyük puntolu yazı; web adresi / e-posta / yayınevi satırları atlanır."""
    best = None
    for pno in range(min(max_pages, len(doc))):
        spans = []
        for b in doc[pno].get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for s in l["spans"]:
                    t = s["text"].strip()
                    if (len(t) < 2 or sum(c.isalpha() for c in t) < 2
                            or re.search(r"www\.|@|\.com|\.org|\.net|yayın|basım|copyright", t, re.I)):
                        continue
                    spans.append((s["size"], t))
        if spans:
            m = max(s for s, _ in spans)
            if best is None or m > best[0]:
                best = (m, " ".join(t for s, t in spans if s >= 0.85 * m))
    if not best:
        return None
    t = re.sub(r"\s+", " ", best[1]).strip(" -–|")
    k = re.search(r"Kitabın Adı:\s*(.+?)\s*Yazar", t)      # künye sayfası: "Kitabın Adı: X Yazar: ..."
    return (k.group(1) if k else t)[:90]


def language(texts):
    s = " ".join(texts)
    letters = [c for c in s if c.isalpha()] or ["x"]
    if sum("؀" <= c <= "ۿ" for c in letters) > 0.4 * len(letters):
        return "fa" if sum(c in "پچژگک" for c in letters) > 0.005 * len(letters) else "ar"
    if sum("Ѐ" <= c <= "ӿ" for c in letters) > 0.4 * len(letters):
        return "bg"
    w = Counter(re.findall(r"\w+", s.lower()))
    marks = {"tr": "ve bir bu için ile olan", "de": "und der die das ist nicht", "en": "the and of is to that",
             "fr": "le la les et est des une", "ku": "û ji ku di bi ên wê"}
    return max(marks, key=lambda k: sum(w[x] for x in marks[k].split()))


def read_book(path):
    st = os.stat(path)
    key = hashlib.md5(("%d|%s|%d|%d" % (VERSION, os.path.abspath(path), st.st_size, int(st.st_mtime))).encode()).hexdigest()[:12]
    cpath = os.path.join(CACHE, "pdf", key + ".json")
    if os.path.exists(cpath):
        return json.load(open(cpath, encoding="utf-8"))
    doc = fitz.open(path)
    chunks = pdf_chunks(doc)
    toc = [{"title": t, "page": p} for lvl, t, p in doc.get_toc() if lvl == 1 and p > 0]
    b = {"id": os.path.splitext(os.path.basename(path))[0], "key": key, "file": path, "pages": len(doc),
         "title": cover_title(doc) or os.path.splitext(os.path.basename(path))[0],
         "lang": language([c["text"] for c in chunks[len(chunks) // 4: len(chunks) // 4 + 40]]),
         "toc": toc if len(toc) >= 3 else [], "chunks": chunks}
    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    json.dump(b, open(cpath, "w", encoding="utf-8"), ensure_ascii=False)
    return b


# ------------------------------------------------------------------ kütüphane
def _shingles(book):
    # yazım farkları (şapkalar, kesme işaretleri) baskılar arasında değişebilir: sadeleştir.
    # içerikle örnekleme (özet % 4 == 0): konumdan bağımsız, metin bir kelime kaysa da aynı diziler seçilir;
    # crc32 belirlenimli (Python'un hash()'i her süreçte farklı tohumla çalışır)
    fold = unicodedata.normalize("NFKD", norm(" ".join(c["text"] for c in book["chunks"])))
    ws = re.findall(r"\w+", "".join(ch for ch in fold if not unicodedata.combining(ch)))
    hs = (zlib.crc32(" ".join(ws[i:i + DUP_SHINGLE]).encode()) for i in range(max(0, len(ws) - DUP_SHINGLE)))
    return {h for h in hs if h % 4 == 0}


def find_duplicates(books):
    """-> {atlanan kitap id: tutulan kitap id}; küçük olan, büyük olanın içinde büyük ölçüde geçiyorsa atlanır."""
    sh = {b["id"]: _shingles(b) for b in books}
    order = sorted(books, key=lambda b: (-len(sh[b["id"]]), b["id"]))
    dup = {}
    for i, small in enumerate(order):
        for big in order[:i]:
            if big["id"] in dup or big["lang"] != small["lang"]:
                continue
            a = sh[small["id"]]
            if a and len(a & sh[big["id"]]) / len(a) > DUP_CONTAIN:
                dup[small["id"]] = big["id"]
                break
    return dup


def patterns(dirs=None):
    """dirs verilmişse onların altındaki tüm PDF'ler, yoksa BOOKS_DIR ortam değişkeni, o da yoksa DEFAULT_PATTERNS."""
    dirs = dirs or (os.environ["BOOKS_DIR"].split(os.pathsep) if os.environ.get("BOOKS_DIR") else None)
    return [os.path.join(d, "**", "*.pdf") for d in dirs] if dirs else DEFAULT_PATTERNS


def load(dirs=None, pats=None):
    """pats: glob kalıpları (verilirse dirs yerine). -> (kitaplar, paragraflar, kütüphane önbellek klasörü, atlananlar)
    kitaplar: [{"id", "title", "lang", "pages", "file", "key", "sections": [{"title", "page", "idx"}]}]
    paragraflar: [{"book": kitap id, "page", "page_end", "text"}]   (idx: bu listedeki sıra)
    atlananlar: [{"id", "reason"}]"""
    pats = pats or patterns(dirs)
    paths = sorted({os.path.realpath(p) for pat in pats for p in glob.glob(pat, recursive=True)})
    assert paths, "PDF bulunamadı: %s" % pats
    books, skipped = [], []
    for p in paths:
        b = read_book(p)
        if len(b["chunks"]) < MIN_CHUNKS:
            skipped.append({"id": b["id"], "reason": "metin çıkmadı (taranmış görüntü olabilir)"})
        else:
            books.append(b)
    known = json.load(open(os.path.join(HERE, "books.json"), encoding="utf-8"))
    for b in books:
        b["title"] = known.get(b["id"], {}).get("title", b["title"])
    dup = find_duplicates(books)
    titles = {b["id"]: b["title"] for b in books}
    skipped += [{"id": k, "reason": "tekrar: %s (%s) içinde geçiyor" % (titles[v], v)} for k, v in dup.items()]
    books = [b for b in books if b["id"] not in dup]

    chunks = []
    for b in books:
        meta = known.get(b["id"], {})
        start = len(chunks)
        chunks += [dict(c, book=b["id"]) for c in b.pop("chunks")]
        b["sections"] = sections(meta.get("sections") or b.pop("toc") or None, chunks, range(start, len(chunks)))
        b.pop("toc", None)
    key = hashlib.md5("|".join(b["key"] for b in books).encode()).hexdigest()[:10]
    cache = os.path.join(CACHE, key)
    os.makedirs(cache, exist_ok=True)
    return books, chunks, cache, skipped


def sections(known, chunks, idx):
    """Bölüm başlangıçları (PDF sayfası) verilmişse onlar, yoksa WINDOW_PAGES'lik sayfa aralıkları."""
    secs = [dict(s, idx=[]) for s in known] if known else []
    for i in idx:
        page = chunks[i]["page"]
        if known:
            j = max([k for k, s in enumerate(secs) if s["page"] <= page] or [0])
        else:
            j = (page - 1) // WINDOW_PAGES
            while len(secs) <= j:
                p = len(secs) * WINDOW_PAGES + 1
                secs.append({"title": "s. %d–%d" % (p, p + WINDOW_PAGES - 1), "page": p, "idx": []})
        secs[j]["idx"].append(i)
    return [s for s in secs if s["idx"]]
