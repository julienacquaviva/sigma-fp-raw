#include "platform.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace sfp {
namespace os {

void sleep_ms(unsigned ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

#ifdef _WIN32

namespace {

std::wstring wide(const std::string& s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(n > 0 ? n : 1, L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    w.resize(n > 0 ? n - 1 : 0);
    return w;
}
std::string utf8(const wchar_t* w) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? n : 1, '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w, -1, &s[0], n, nullptr, nullptr);
    s.resize(n > 0 ? n - 1 : 0);
    return s;
}
uint64_t u64(DWORD hi, DWORD lo) { return (static_cast<uint64_t>(hi) << 32) | lo; }

}  // namespace

std::string env(const char* name) {
    char buf[1024];
    DWORD n = GetEnvironmentVariableA(name, buf, sizeof buf);
    return n > 0 && n < sizeof buf ? std::string(buf, n) : std::string();
}

bool stat_file(const std::string& path, FileInfo& info) {
    WIN32_FILE_ATTRIBUTE_DATA a;
    if (!GetFileAttributesExW(wide(path).c_str(), GetFileExInfoStandard, &a)) return false;
    info.size = u64(a.nFileSizeHigh, a.nFileSizeLow);
    info.time = u64(a.ftLastWriteTime.dwHighDateTime, a.ftLastWriteTime.dwLowDateTime);
    return true;
}

bool exists(const std::string& path) { return GetFileAttributesW(wide(path).c_str()) != INVALID_FILE_ATTRIBUTES; }

std::vector<FileInfo> list_matching(const std::string& pathPrefix, const std::string& suffix) {
    std::vector<FileInfo> out;
    WIN32_FIND_DATAW fd;
    HANDLE hf = FindFirstFileExW(wide(pathPrefix + "*" + suffix).c_str(), FindExInfoBasic, &fd, FindExSearchNameMatch, nullptr, FIND_FIRST_EX_LARGE_FETCH);
    if (hf == INVALID_HANDLE_VALUE) return out;
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        FileInfo f;
        f.name = utf8(fd.cFileName);
        // The pattern also matches 8.3 short names: keep real matches of the suffix only.
        if (f.name.size() < suffix.size()) continue;
        bool same = true;
        for (size_t i = 0; i < suffix.size() && same; ++i)
            same = std::toupper(static_cast<unsigned char>(f.name[f.name.size() - suffix.size() + i])) == std::toupper(static_cast<unsigned char>(suffix[i]));
        if (!same) continue;
        f.size = u64(fd.nFileSizeHigh, fd.nFileSizeLow);
        f.time = u64(fd.ftLastWriteTime.dwHighDateTime, fd.ftLastWriteTime.dwLowDateTime);
        out.push_back(std::move(f));
    } while (FindNextFileW(hf, &fd));
    FindClose(hf);
    return out;
}

bool make_dir(const std::string& dir) {
    return CreateDirectoryW(wide(dir).c_str(), nullptr) || GetLastError() == ERROR_ALREADY_EXISTS;
}

bool write_file(const std::string& path, const void* data, size_t size) {
    HANDLE h = CreateFileW(wide(path).c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    size_t done = 0;
    bool ok = true;
    while (done < size && ok) {
        DWORD put = 0;
        ok = WriteFile(h, static_cast<const char*>(data) + done, static_cast<DWORD>(std::min<size_t>(size - done, 16u << 20)), &put, nullptr) && put > 0;
        done += put;
    }
    CloseHandle(h);
    return ok;
}

bool replace_file(const std::string& from, const std::string& to) {
    return MoveFileExW(wide(from).c_str(), wide(to).c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
}

bool remove_file(const std::string& path) { return DeleteFileW(wide(path).c_str()) != 0; }

std::string user_cache_dir() { return env("LOCALAPPDATA"); }

File::File(const std::string& path) {
    h = CreateFileW(wide(path).c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, nullptr);
}
File::~File() { if (h != INVALID_HANDLE_VALUE) CloseHandle(h); }
bool File::ok() const { return h != INVALID_HANDLE_VALUE; }
size_t File::read(long long offset, void* out, size_t n) const {
    LARGE_INTEGER li;
    li.QuadPart = offset;
    if (!SetFilePointerEx(h, li, nullptr, FILE_BEGIN)) return 0;
    size_t done = 0;
    while (done < n) {
        DWORD got = 0;
        if (!ReadFile(h, static_cast<char*>(out) + done, static_cast<DWORD>(n - done), &got, nullptr) || !got) break;
        done += got;
    }
    return done;
}

void* load_library(const char* name) { return LoadLibraryA(name); }
void* library_symbol(void* library, const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library), name));
}

#else  // macOS, Linux

namespace {

bool same_nocase(const char* a, const char* b, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (std::toupper(static_cast<unsigned char>(a[i])) != std::toupper(static_cast<unsigned char>(b[i]))) return false;
    return true;
}
uint64_t mtime(const struct stat& st) {
#ifdef __APPLE__
    return static_cast<uint64_t>(st.st_mtimespec.tv_sec) * 1000000000ull + static_cast<uint64_t>(st.st_mtimespec.tv_nsec);
#else
    return static_cast<uint64_t>(st.st_mtim.tv_sec) * 1000000000ull + static_cast<uint64_t>(st.st_mtim.tv_nsec);
#endif
}

}  // namespace

std::string env(const char* name) {
    const char* v = std::getenv(name);
    return v ? v : "";
}

bool stat_file(const std::string& path, FileInfo& info) {
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) return false;
    info.size = static_cast<uint64_t>(st.st_size);
    info.time = mtime(st);
    return true;
}

bool exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

std::vector<FileInfo> list_matching(const std::string& pathPrefix, const std::string& suffix) {
    std::vector<FileInfo> out;
    const size_t slash = pathPrefix.find_last_of('/');
    const std::string dir = slash == std::string::npos ? std::string(".") : (slash == 0 ? std::string("/") : pathPrefix.substr(0, slash));
    const std::string stem = slash == std::string::npos ? pathPrefix : pathPrefix.substr(slash + 1);
    DIR* d = ::opendir(dir.c_str());
    if (!d) return out;
    while (struct dirent* e = ::readdir(d)) {
        const std::string name = e->d_name;
        if (name.size() < stem.size() + suffix.size()) continue;
        if (!same_nocase(name.c_str(), stem.c_str(), stem.size())) continue;
        if (!same_nocase(name.c_str() + name.size() - suffix.size(), suffix.c_str(), suffix.size())) continue;
        struct stat st;
        if (::stat((dir + "/" + name).c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;
        FileInfo f;
        f.name = name;
        f.size = static_cast<uint64_t>(st.st_size);
        f.time = mtime(st);
        out.push_back(std::move(f));
    }
    ::closedir(d);
    return out;
}

bool make_dir(const std::string& dir) {
    if (::mkdir(dir.c_str(), 0755) == 0) return true;
    struct stat st;
    return ::stat(dir.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool write_file(const std::string& path, const void* data, size_t size) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = std::fwrite(data, 1, size, f) == size;
    ok = std::fclose(f) == 0 && ok;
    return ok;
}

bool replace_file(const std::string& from, const std::string& to) { return ::rename(from.c_str(), to.c_str()) == 0; }

bool remove_file(const std::string& path) { return ::unlink(path.c_str()) == 0; }

std::string user_cache_dir() {
    const std::string home = env("HOME");
#ifdef __APPLE__
    if (home.empty()) return "";
    return home + "/Library/Caches";
#else
    const std::string xdg = env("XDG_CACHE_HOME");
    if (!xdg.empty()) return xdg;
    if (home.empty()) return "";
    make_dir(home + "/.cache");
    return home + "/.cache";
#endif
}

File::File(const std::string& path) { fd = ::open(path.c_str(), O_RDONLY); }
File::~File() { if (fd >= 0) ::close(fd); }
bool File::ok() const { return fd >= 0; }
size_t File::read(long long offset, void* out, size_t n) const {
    size_t done = 0;
    while (done < n) {
        ssize_t got = ::pread(fd, static_cast<char*>(out) + done, n - done, static_cast<off_t>(offset + static_cast<long long>(done)));
        if (got <= 0) break;
        done += static_cast<size_t>(got);
    }
    return done;
}

void* load_library(const char* name) { return ::dlopen(name, RTLD_NOW | RTLD_LOCAL); }
void* library_symbol(void* library, const char* name) { return ::dlsym(library, name); }

#endif

}  // namespace os
}  // namespace sfp
