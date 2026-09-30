#include "mogg.hpp"

#include <cstring>
#include <fstream>
#include <limits>

#if defined(__APPLE__) || defined(__unix__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <stdexcept>

namespace moge {
namespace {

constexpr size_t kMoggHeaderBytes = 40;
constexpr uint32_t kMaxMetadataEntries = 1u << 20;
constexpr uint32_t kMaxTensorEntries = 1u << 20;
constexpr uint32_t kMaxMetadataPayloadBytes = 16u * 1024u * 1024u;
constexpr uint32_t kMaxMetadataListEntries = 1u << 20;
constexpr uint32_t kMaxMetadataStringBytes = 1u * 1024u * 1024u;
constexpr uint32_t kMaxAlignment = 1u << 20;

struct Reader {
    const uint8_t * p = nullptr;
    const uint8_t * end = nullptr;

    size_t remaining() const {
        if (!p || !end || p > end) return 0;
        return static_cast<size_t>(end - p);
    }
    void need(size_t n) const {
        if (n > remaining()) throw std::runtime_error("truncated MOGG file");
    }
    template<class T> T pod() {
        need(sizeof(T)); T v; std::memcpy(&v, p, sizeof(T)); p += sizeof(T); return v;
    }
    std::string str16() {
        uint16_t n = pod<uint16_t>(); need(n);
        std::string s(reinterpret_cast<const char *>(p), n); p += n; return s;
    }
};

constexpr uint8_t MAGIC[8] = {'M','O','G','G',0,0,0,1};
constexpr uint32_t VERSION = 1;

enum : uint8_t { META_I64=1, META_F64=2, META_STRING=3, META_I64_LIST=4, META_STRING_LIST=5, META_BOOL=6 };

size_t checked_mul_size(size_t a, size_t b, const char * what) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
        throw std::runtime_error(std::string("MOGG ") + what + " size overflow");
    return a * b;
}

bool valid_mogg_type(MoggType t) {
    switch (t) {
        case MoggType::F32:
        case MoggType::F16:
        case MoggType::BF16:
        case MoggType::I32:
        case MoggType::Q4_K:
        case MoggType::Q6_K:
        case MoggType::Q8_0:
        case MoggType::IQ4_XS:
            return true;
        default:
            return false;
    }
}

uint64_t expected_tensor_bytes(const TensorInfo & t) {
    uint64_t rows = 1;
    for (int d = 1; d < t.ndim; ++d) {
        const uint64_t n = static_cast<uint64_t>(t.ne[d]);
        if (rows > std::numeric_limits<uint64_t>::max() / n) throw std::runtime_error("MOGG tensor element count overflow for " + t.name);
        rows *= n;
    }
    const uint64_t ne0 = static_cast<uint64_t>(t.ne[0]);
    uint64_t row_bytes = 0;
    switch (t.type) {
        case MoggType::F32:
        case MoggType::I32:
            if (ne0 > std::numeric_limits<uint64_t>::max() / 4u) throw std::runtime_error("MOGG tensor row size overflow for " + t.name);
            row_bytes = ne0 * 4u;
            break;
        case MoggType::F16:
        case MoggType::BF16:
            if (ne0 > std::numeric_limits<uint64_t>::max() / 2u) throw std::runtime_error("MOGG tensor row size overflow for " + t.name);
            row_bytes = ne0 * 2u;
            break;
        case MoggType::Q8_0:
            if (ne0 % 32u) throw std::runtime_error("bad Q8_0 row width for " + t.name);
            row_bytes = (ne0 / 32u) * 34u;
            break;
        case MoggType::Q4_K:
            if (ne0 % 256u) throw std::runtime_error("bad Q4_K row width for " + t.name);
            row_bytes = (ne0 / 256u) * 144u;
            break;
        case MoggType::Q6_K:
            if (ne0 % 256u) throw std::runtime_error("bad Q6_K row width for " + t.name);
            row_bytes = (ne0 / 256u) * 210u;
            break;
        case MoggType::IQ4_XS:
            if (ne0 % 256u) throw std::runtime_error("bad IQ4_XS row width for " + t.name);
            row_bytes = (ne0 / 256u) * 136u;
            break;
        default:
            throw std::runtime_error("unknown MOGG tensor type for " + t.name);
    }
    if (rows != 0 && row_bytes > std::numeric_limits<uint64_t>::max() / rows)
        throw std::runtime_error("MOGG tensor byte size overflow for " + t.name);
    return row_bytes * rows;
}

MetaValue read_meta_value(Reader & r, uint8_t tag, uint32_t bytes) {
    if (bytes > kMaxMetadataPayloadBytes) throw std::runtime_error("MOGG metadata payload too large");
    const uint8_t * start = r.p;
    r.need(bytes);
    MetaValue out;
    switch (tag) {
        case META_I64:
            if (bytes != sizeof(int64_t)) throw std::runtime_error("bad integer metadata payload length");
            out = r.pod<int64_t>();
            break;
        case META_F64:
            if (bytes != sizeof(double)) throw std::runtime_error("bad float metadata payload length");
            out = r.pod<double>();
            break;
        case META_STRING: {
            if (bytes < sizeof(uint32_t)) throw std::runtime_error("bad string metadata payload");
            uint32_t n = r.pod<uint32_t>();
            if (n > kMaxMetadataStringBytes || n != bytes - sizeof(uint32_t)) throw std::runtime_error("bad string metadata length");
            r.need(n);
            out = std::string(reinterpret_cast<const char *>(r.p), n); r.p += n; break;
        }
        case META_I64_LIST: {
            if (bytes < sizeof(uint32_t)) throw std::runtime_error("bad integer-list metadata payload");
            uint32_t n = r.pod<uint32_t>();
            if (n > kMaxMetadataListEntries || static_cast<uint64_t>(n) * sizeof(int64_t) != bytes - sizeof(uint32_t))
                throw std::runtime_error("bad integer-list metadata length");
            std::vector<int64_t> v; v.reserve(n);
            for (uint32_t i=0;i<n;++i) v.push_back(r.pod<int64_t>());
            out = std::move(v); break;
        }
        case META_STRING_LIST: {
            if (bytes < sizeof(uint32_t)) throw std::runtime_error("bad string-list metadata payload");
            uint32_t n = r.pod<uint32_t>();
            if (n > kMaxMetadataListEntries || static_cast<uint64_t>(n) * 2u > bytes - sizeof(uint32_t))
                throw std::runtime_error("bad string-list metadata length");
            std::vector<std::string> v; v.reserve(n);
            for (uint32_t i=0;i<n;++i) {
                std::string s = r.str16();
                if (s.size() > kMaxMetadataStringBytes) throw std::runtime_error("MOGG metadata string too large");
                v.push_back(std::move(s));
            }
            out = std::move(v); break;
        }
        case META_BOOL:
            if (bytes != 1) throw std::runtime_error("bad boolean metadata payload length");
            out = static_cast<bool>(r.pod<uint8_t>()); break;
        default: throw std::runtime_error("unknown MOGG metadata type " + std::to_string(tag));
    }
    if (static_cast<uint32_t>(r.p - start) != bytes) throw std::runtime_error("bad MOGG metadata payload length");
    return out;
}

} // namespace

MoggFile::MoggFile(const std::filesystem::path & path) {
#if defined(__APPLE__) || defined(__unix__)
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open " + path.string());
    struct stat st{};
    if (::fstat(fd_, &st) != 0 || st.st_size < static_cast<off_t>(kMoggHeaderBytes)) {
        ::close(fd_); fd_ = -1;
        throw std::runtime_error("MOGG file too small");
    }
    if (st.st_size < 0 || static_cast<uintmax_t>(st.st_size) > static_cast<uintmax_t>(std::numeric_limits<size_t>::max())) {
        ::close(fd_); fd_ = -1;
        throw std::runtime_error("MOGG file is too large for this process");
    }
    size_ = static_cast<size_t>(st.st_size);
    // MAP_PRIVATE + PROT_WRITE does not write the file; it only permits a
    // backend API that accepts void*. Weight tensors remain logically const.
    void * mp = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd_, 0);
    if (mp != MAP_FAILED) {
        data_ = static_cast<uint8_t *>(mp);
        mapped_ = true;
    } else {
        ::close(fd_); fd_ = -1;
    }
#endif
    if (!data_) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("cannot open " + path.string());
        const auto n = f.tellg();
        if (n < static_cast<std::streamoff>(kMoggHeaderBytes)) throw std::runtime_error("MOGG file too small");
        const auto un = static_cast<uintmax_t>(n);
        if (un > static_cast<uintmax_t>(std::numeric_limits<size_t>::max())) throw std::runtime_error("MOGG file is too large for this process");
        bytes_.resize(static_cast<size_t>(un));
        f.seekg(0);
        if (!f.read(reinterpret_cast<char *>(bytes_.data()), static_cast<std::streamsize>(bytes_.size())))
            throw std::runtime_error("failed reading " + path.string());
        data_ = bytes_.data();
        size_ = bytes_.size();
    }

    parse();
}

MoggFile::MoggFile(const void * data, size_t size, bool copy) {
    if (!data || size < kMoggHeaderBytes) throw std::runtime_error("MOGG buffer too small");
    size_ = size;
    if (copy) {
#if defined(__APPLE__) || defined(__unix__)
        // Anonymous mmap keeps owned memory page-backed/aligned, allowing the
        // same ggml/Metal host-pointer wrapping path used by file-backed mmap.
        void * mp = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mp != MAP_FAILED) {
            data_ = static_cast<uint8_t *>(mp);
            mapped_ = true;
            std::memcpy(data_, data, size_);
        }
#endif
        if (!data_) {
            bytes_.resize(size_);
            std::memcpy(bytes_.data(), data, size_);
            data_ = bytes_.data();
        }
    } else {
        // The parser and runtime treat model bytes as immutable. The pointer is
        // stored as writable only because some ggml backend APIs accept void*.
        data_ = const_cast<uint8_t *>(static_cast<const uint8_t *>(data));
    }
    parse();
}

void MoggFile::parse() {
    Reader h{data_, data_ + size_};
    h.need(8);
    if (std::memcmp(h.p, MAGIC, 8) != 0) throw std::runtime_error("not a MOGG v1 file");
    h.p += 8;
    const uint32_t version = h.pod<uint32_t>();
    const uint32_t n_meta = h.pod<uint32_t>();
    const uint32_t n_tensors = h.pod<uint32_t>();
    const uint32_t alignment = h.pod<uint32_t>();
    const uint64_t meta_offset = h.pod<uint64_t>();
    const uint64_t data_start = h.pod<uint64_t>();
    if (version != VERSION) throw std::runtime_error("unsupported MOGG version");
    if (alignment < 64 || alignment > kMaxAlignment || (alignment & (alignment - 1)) != 0) throw std::runtime_error("bad MOGG alignment");
    if (meta_offset < kMoggHeaderBytes || meta_offset >= data_start || data_start > size_) throw std::runtime_error("bad MOGG offsets");
    if ((data_start % alignment) != 0) throw std::runtime_error("unaligned MOGG data section");

    const size_t directory_bytes = static_cast<size_t>(data_start - meta_offset);
    // Empty-key metadata has an 8-byte minimum record (u16 length, tag,
    // u32 payload length, one-byte bool). Tensor records have a 54-byte
    // minimum. Bound counts before reserve()/hash insertion to avoid memory
    // exhaustion from hostile headers.
    if (n_meta > kMaxMetadataEntries || static_cast<uint64_t>(n_meta) * 8u > directory_bytes)
        throw std::runtime_error("unreasonable MOGG metadata count");
    if (n_tensors > kMaxTensorEntries || static_cast<uint64_t>(n_tensors) * 54u > directory_bytes)
        throw std::runtime_error("unreasonable MOGG tensor count");

    Reader r{data_ + meta_offset, data_ + data_start};
    for (uint32_t i=0; i<n_meta; ++i) {
        std::string key = r.str16();
        if (key.empty()) throw std::runtime_error("empty MOGG metadata key");
        uint8_t tag = r.pod<uint8_t>();
        uint32_t payload = r.pod<uint32_t>();
        auto [it, inserted] = meta_.emplace(std::move(key), read_meta_value(r, tag, payload));
        if (!inserted) throw std::runtime_error("duplicate MOGG metadata key");
    }
    tensors_.reserve(n_tensors);
    for (uint32_t i=0; i<n_tensors; ++i) {
        TensorInfo t;
        t.name = r.str16();
        if (t.name.empty()) throw std::runtime_error("empty MOGG tensor name");
        t.type = static_cast<MoggType>(r.pod<uint16_t>());
        t.ndim = r.pod<uint8_t>();
        t.flags = r.pod<uint8_t>();
        if (!valid_mogg_type(t.type)) throw std::runtime_error("unknown tensor type for " + t.name);
        if (t.ndim < 1 || t.ndim > 4) throw std::runtime_error("bad ndim for " + t.name);
        for (int d=0; d<4; ++d) t.ne[d] = r.pod<int64_t>();
        t.offset = r.pod<uint64_t>();
        t.size = r.pod<uint64_t>();
        if (t.offset < data_start || t.offset > size_ || t.size > size_ - t.offset)
            throw std::runtime_error("bad tensor extent for " + t.name);
        if ((t.offset % alignment) != 0) throw std::runtime_error("unaligned tensor payload for " + t.name);
        for (int d=0; d<t.ndim; ++d) if (t.ne[d] <= 0) throw std::runtime_error("bad tensor shape for " + t.name);
        for (int d=t.ndim; d<4; ++d) if (t.ne[d] != 1) throw std::runtime_error("bad unused tensor dimension for " + t.name);
        const uint64_t expected = expected_tensor_bytes(t);
        if (t.size != expected) throw std::runtime_error("tensor payload size does not match shape/type for " + t.name);
        if (tensor_index_.find(t.name) != tensor_index_.end()) throw std::runtime_error("duplicate MOGG tensor name: " + t.name);
        tensor_index_[t.name] = tensors_.size();
        tensors_.push_back(std::move(t));
    }
    // Padding between the directory and aligned data section must be zero or
    // absent. Do not require a specific padding length, but never parse it as
    // metadata/tensors.
}

MoggFile::~MoggFile() {
#if defined(__APPLE__) || defined(__unix__)
    if (mapped_ && data_ && size_) ::munmap(data_, size_);
    if (fd_ >= 0) ::close(fd_);
#endif
    data_ = nullptr;
    size_ = 0;
}

const TensorInfo * MoggFile::find_tensor(const std::string & name) const {
    auto it = tensor_index_.find(name);
    return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

const uint8_t * MoggFile::tensor_data(const TensorInfo & t) const {
    if (t.offset > size_ || t.size > size_ - t.offset) throw std::runtime_error("invalid tensor extent");
    return data_ + t.offset;
}

} // namespace moge
