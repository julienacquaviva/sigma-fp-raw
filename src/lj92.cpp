// Lossless JPEG decoder (SOF3, Huffman, predictors 1-7, restart intervals).
// Strict about the frame: every declared sample must decode; trailing padding
// after the last sample (the fp hardware pads tiles with zero bytes) is ignored.
#include "lj92.h"

#include <cstring>
#include <vector>

namespace sfp {
namespace {

constexpr int kLutBits = 12;

struct Huffman {
    bool present = false;
    uint16_t lut[1 << kLutBits];   // (len << 8) | symbol, 0 = longer than kLutBits
    int maxcode[18];
    int valptr[17];
    int mincode[17];
    uint8_t symbols[256];
};

bool build(Huffman& h, const uint8_t* counts, const uint8_t* syms, int nsyms) {
    std::memcpy(h.symbols, syms, nsyms);
    std::memset(h.lut, 0, sizeof h.lut);
    int code = 0, k = 0;
    for (int len = 1; len <= 16; ++len) {
        h.valptr[len] = k;
        h.mincode[len] = code;
        for (int i = 0; i < counts[len - 1]; ++i, ++k, ++code) {
            if (len <= kLutBits) {
                int shift = kLutBits - len;
                for (int f = 0; f < (1 << shift); ++f)
                    h.lut[(code << shift) | f] = static_cast<uint16_t>((len << 8) | syms[k]);
            }
        }
        h.maxcode[len] = counts[len - 1] ? code - 1 : -1;
        if (code > (1 << len)) return false;
        code <<= 1;
    }
    h.maxcode[17] = 0x7FFFFFFF;
    h.present = true;
    return true;
}

struct Bits {
    const uint8_t* p;
    const uint8_t* end;
    uint64_t buf = 0;
    int n = 0;
    bool marker = false;   // hit a marker: feed zeros

    void fill() {
        while (n <= 56) {
            uint32_t byte = 0;
            if (!marker && p < end) {
                byte = *p;
                if (byte == 0xFF) {
                    uint8_t next = p + 1 < end ? p[1] : 0xD9;
                    if (next == 0x00) p += 2;
                    else { marker = true; byte = 0; }
                } else {
                    ++p;
                }
            }
            buf |= static_cast<uint64_t>(byte) << (56 - n);
            n += 8;
        }
    }
    inline uint32_t peek(int k) { if (n < k) fill(); return static_cast<uint32_t>(buf >> (64 - k)); }
    inline void skip(int k) { buf <<= k; n -= k; }
    inline uint32_t get(int k) { uint32_t v = peek(k); skip(k); return v; }
    // Byte-align and consume an RSTn marker.
    bool restart() {
        buf = 0; n = 0;
        if (!marker) {
            // Skip fill bytes to the marker.
            while (p < end && *p != 0xFF) ++p;
        }
        marker = false;
        while (p + 1 < end && p[0] == 0xFF && p[1] == 0xFF) ++p;
        if (p + 1 < end && p[0] == 0xFF && p[1] >= 0xD0 && p[1] <= 0xD7) { p += 2; return true; }
        return false;
    }
};

inline int decode_diff(Bits& b, const Huffman& h) {
    uint32_t look = b.peek(kLutBits);
    uint32_t e = h.lut[look];
    int len, s;
    if (e) {
        len = e >> 8;
        s = e & 0xFF;
        b.skip(len);
    } else {
        uint32_t code = b.get(kLutBits);
        len = kLutBits;
        while (len < 16 && static_cast<int>(code) > h.maxcode[len]) {
            code = (code << 1) | b.get(1);
            ++len;
        }
        if (static_cast<int>(code) > h.maxcode[len]) return 0x40000;   // invalid code
        s = h.symbols[h.valptr[len] + static_cast<int>(code) - h.mincode[len]];
    }
    if (s == 0) return 0;
    if (s == 16) return 32768;
    if (s > 16) return 0x40000;
    int v = static_cast<int>(b.get(s));
    if (v < (1 << (s - 1))) v -= (1 << s) - 1;
    return v;
}

}  // namespace

bool lj92_decode(const uint8_t* data, size_t size, uint16_t* out, int stride, int outW, int outH,
                 int expectW, int expectH, std::string& error) {
    const uint8_t* p = data;
    const uint8_t* end = data + size;
    if (size < 4 || p[0] != 0xFF || p[1] != 0xD8) { error = "LJ92: no SOI"; return false; }
    p += 2;
    std::vector<Huffman> tables(4);
    int precision = 0, X = 0, Y = 0, nf = 0, restartInterval = 0;
    int compId[4] = {}, td[4] = {};
    int predictor = 1, pt = 0, ns = 0;
    int scanComp[4] = {};
    for (;;) {
        while (p < end && *p != 0xFF) ++p;      // tolerate fill
        while (p + 1 < end && p[1] == 0xFF) ++p;
        if (p + 4 > end) { error = "LJ92: truncated header"; return false; }
        uint8_t m = p[1];
        p += 2;
        if (m == 0xD8 || (m >= 0xD0 && m <= 0xD7) || m == 0x01) continue;
        if (m == 0xD9) { error = "LJ92: EOI before scan"; return false; }
        int len = (p[0] << 8) | p[1];
        if (len < 2 || p + len > end) { error = "LJ92: bad segment length"; return false; }
        const uint8_t* s = p + 2;
        const uint8_t* se = p + len;
        if (m == 0xC4) {
            while (s + 17 <= se) {
                int tc = s[0] >> 4, th = s[0] & 15;
                int total = 0;
                for (int i = 0; i < 16; ++i) total += s[1 + i];
                if (tc != 0 || th > 3 || total > 256 || s + 17 + total > se) { error = "LJ92: bad DHT"; return false; }
                if (!build(tables[th], s + 1, s + 17, total)) { error = "LJ92: bad Huffman table"; return false; }
                s += 17 + total;
            }
        } else if (m == 0xC3) {
            precision = s[0];
            Y = (s[1] << 8) | s[2];
            X = (s[3] << 8) | s[4];
            nf = s[5];
            if (nf < 1 || nf > 4 || precision < 2 || precision > 16) { error = "LJ92: bad SOF3"; return false; }
            for (int i = 0; i < nf; ++i) {
                compId[i] = s[6 + 3 * i];
                if (s[7 + 3 * i] != 0x11) { error = "LJ92: subsampling unsupported"; return false; }
            }
        } else if (m >= 0xC0 && m <= 0xCF && m != 0xC4 && m != 0xC8 && m != 0xCC) {
            error = "LJ92: not a lossless (SOF3) stream";
            return false;
        } else if (m == 0xDD) {
            restartInterval = (s[0] << 8) | s[1];
        } else if (m == 0xDA) {
            ns = s[0];
            if (ns != nf || !nf) { error = "LJ92: non-interleaved scan unsupported"; return false; }
            for (int i = 0; i < ns; ++i) {
                int cs = s[1 + 2 * i];
                int k = 0;
                while (k < nf && compId[k] != cs) ++k;
                if (k == nf) { error = "LJ92: scan component unknown"; return false; }
                scanComp[i] = k;
                td[k] = s[2 + 2 * i] >> 4;
                if (td[k] > 3 || !tables[td[k]].present) { error = "LJ92: missing Huffman table"; return false; }
            }
            predictor = s[1 + 2 * ns];
            pt = s[3 + 2 * ns] & 15;
            if (predictor < 1 || predictor > 7) { error = "LJ92: bad predictor"; return false; }
            p = se;
            break;
        }
        p = se;
    }
    const int W = X * nf;
    if (W != expectW || Y != expectH) { error = "LJ92: tile size mismatch"; return false; }
    (void)scanComp;

    Bits b{p, end};
    std::vector<uint16_t> prev(W), cur(W);
    const int initial = 1 << (precision - pt - 1);
    const Huffman* ht[4];
    for (int c = 0; c < nf; ++c) ht[c] = &tables[td[c]];
    int mcus = 0;
    bool firstRow = true;
    for (int y = 0; y < Y; ++y) {
        for (int x = 0; x < X; ++x) {
            if (restartInterval && mcus == restartInterval) {
                if (!b.restart()) { error = "LJ92: missing restart marker"; return false; }
                mcus = 0;
                firstRow = true;
            }
            for (int c = 0; c < nf; ++c) {
                int i = x * nf + c;
                int pred;
                if (firstRow) pred = x == 0 ? initial : cur[i - nf];
                else if (x == 0) pred = prev[i];
                else {
                    int ra = cur[i - nf], rb = prev[i], rc = prev[i - nf];
                    switch (predictor) {
                        case 1: pred = ra; break;
                        case 2: pred = rb; break;
                        case 3: pred = rc; break;
                        case 4: pred = ra + rb - rc; break;
                        case 5: pred = ra + ((rb - rc) >> 1); break;
                        case 6: pred = rb + ((ra - rc) >> 1); break;
                        default: pred = (ra + rb) >> 1; break;
                    }
                }
                int d = decode_diff(b, *ht[c]);
                if (d == 0x40000) { error = "LJ92: invalid Huffman code"; return false; }
                cur[i] = static_cast<uint16_t>(pred + d);
            }
            ++mcus;
        }
        if (b.marker && b.n <= 0) { error = "LJ92: stream ended early"; return false; }
        if (y < outH) {
            uint16_t* row = out + static_cast<size_t>(y) * stride;
            int n = W < outW ? W : outW;
            if (pt) for (int i = 0; i < n; ++i) row[i] = static_cast<uint16_t>(cur[i] << pt);
            else std::memcpy(row, cur.data(), n * sizeof(uint16_t));
        }
        std::swap(prev, cur);
        firstRow = false;
    }
    return true;
}

}  // namespace sfp
