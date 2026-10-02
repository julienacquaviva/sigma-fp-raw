// Lossless JPEG (ITU T.81 process 14, SOF3) decoder for DNG tiles.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace sfp {

// Decodes one LJ92 stream into a W x H sample tile (W = jpegWidth * components),
// writing rows of `outW` samples (clipped to outW x outH) at `out` with row stride
// `stride` samples. Samples beyond the stream's end or tile padding are ignored;
// the stream must supply every sample of the declared frame.
bool lj92_decode(const uint8_t* data, size_t size, uint16_t* out, int stride,
                 int outW, int outH, int expectW, int expectH, std::string& error);

}  // namespace sfp
