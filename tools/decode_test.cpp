// Decode DNG raw to a .u16 file for comparison with tifffile; prints timing.
#include <chrono>
#include <cstdio>
#include "../src/dng.h"
int main(int argc, char** argv) {
    if (argc < 3) return 2;
    std::vector<uint8_t> d; std::string e; sfp::DngInfo info;
    if (!sfp::read_file(argv[1], d, e) || !sfp::parse_dng(d.data(), d.size(), info, e)) { std::printf("ERR %s\n", e.c_str()); return 1; }
    std::vector<uint16_t> out((size_t)info.width * info.height);
    int threads = argc > 3 ? atoi(argv[3]) : 1;
    auto t0 = std::chrono::steady_clock::now();
    int reps = argc > 4 ? atoi(argv[4]) : 1;
    for (int i = 0; i < reps; ++i)
        if (!sfp::decode_raw(d.data(), d.size(), info, out.data(), threads, e)) { std::printf("ERR %s\n", e.c_str()); return 1; }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    FILE* f = std::fopen(argv[2], "wb"); std::fwrite(out.data(), 2, out.size(), f); std::fclose(f);
    std::printf("%dx%d comp %d tiles %zu black %.1f %.1f %.1f %.1f white %.0f crop %d,%d %dx%d active %d,%d iso %d fps %.3f BE %.2f neutral %.4f %.4f %.4f illum %d/%d decode %.2f ms (%d threads)\n",
        info.width, info.height, info.compression, info.offsets.size(), info.black[0], info.black[1], info.black[2], info.black[3], info.white,
        info.cropX, info.cropY, info.cropW, info.cropH, info.activeLeft, info.activeTop, info.iso, info.fps, info.baselineExposure,
        info.neutral[0], info.neutral[1], info.neutral[2], info.illum1, info.illum2, ms, threads);
    return 0;
}
