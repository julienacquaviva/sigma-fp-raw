// The few operating-system services the plug-in needs (Windows, macOS, Linux). Paths are UTF-8.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sfp {
namespace os {

#ifdef _WIN32
const char kSep = '\\';
#else
const char kSep = '/';
#endif

struct FileInfo {
    std::string name;      // without the folder
    uint64_t size = 0;
    uint64_t time = 0;     // last write; only compared, the unit differs between systems
};

// Environment variable, empty when unset.
std::string env(const char* name);

bool stat_file(const std::string& path, FileInfo& info);
bool exists(const std::string& path);

// Files (not folders) named <pathPrefix><anything><suffix>; pathPrefix carries the folder and
// may end in a separator. Names are compared without regard to ASCII case.
std::vector<FileInfo> list_matching(const std::string& pathPrefix, const std::string& suffix);

bool make_dir(const std::string& dir);
bool write_file(const std::string& path, const void* data, size_t size);
bool replace_file(const std::string& from, const std::string& to);
bool remove_file(const std::string& path);

// Per-user folder for cache files ("" when the system gives none): %LOCALAPPDATA%,
// ~/Library/Caches, $XDG_CACHE_HOME or ~/.cache.
std::string user_cache_dir();

// Read access at any offset.
class File {
public:
    explicit File(const std::string& path);
    ~File();
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    bool ok() const;
    // Reads up to n bytes at an offset; returns the number read.
    size_t read(long long offset, void* out, size_t n) const;

private:
#ifdef _WIN32
    void* h;
#else
    int fd;
#endif
};

void sleep_ms(unsigned ms);

// Shared library by file name; nullptr when it is not installed.
void* load_library(const char* name);
void* library_symbol(void* library, const char* name);

}  // namespace os
}  // namespace sfp
