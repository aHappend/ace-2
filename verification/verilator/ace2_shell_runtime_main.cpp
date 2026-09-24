#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <cfenv>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "Vace2_shell_runtime_harness.h"
#include "ace2_runtime_identity_profile.h"
#include "verilated.h"

namespace fs = std::filesystem;

namespace {

constexpr std::uint32_t kPackageVersion1 = 1;
constexpr std::uint32_t kPackageVersion2 = 2;
constexpr std::uint32_t kJournalVersion1 = 1;
constexpr std::uint32_t kJournalVersion2 = 2;
constexpr std::uint32_t kPackageV2HeaderBytes = 216;
constexpr std::uint32_t kPackageV2DynamicExtensionBytes = 64;
constexpr std::uint32_t kPackageV2DynamicRecordBytes = 72;
constexpr std::uint32_t kDynamicScale32FeatureSidecars = 1;
constexpr std::uint8_t kDynamicScale32Flag = 1U << 6;
constexpr std::uint8_t kDynamicScale32HostPlanLayer = 0xffU;
constexpr std::uint8_t kDynamicScale32HostPlanOpcode = 0xfeU;
constexpr std::uint8_t kDynamicScale32InitialGroupLanes = 128U;
constexpr std::uint8_t kDynamicScale32InitialGroupCount = 7U;
constexpr std::uint32_t kDynamicScale32InitialElements = 896U;
constexpr std::uint8_t kDynamicScale32FrozenFlags = kDynamicScale32Flag | 0x09U;
constexpr std::uint64_t kDynamicScale32RmsSrc0 = 0x0000001000000000ULL;
constexpr std::uint64_t kDynamicScale32RmsSrc1 = 0x0000000000000000ULL;
constexpr std::uint64_t kDynamicScale32RmsDst = 0x0000001000000700ULL;
constexpr std::uint64_t kDynamicScale32RmsScale = 0x0000000300000000ULL;
constexpr std::uint64_t kDynamicScale32RmsScratch = 0x0000000000000000ULL;
constexpr std::uint64_t kDynamicScale32QSrc0 = 0x0000001000000700ULL;
constexpr std::uint64_t kDynamicScale32QSrc1 = 0x0000000100000000ULL;
constexpr std::uint64_t kDynamicScale32QDst = 0x0000001000000a80ULL;
constexpr std::uint64_t kDynamicScale32QScale = 0x0000000200000000ULL;
constexpr std::uint64_t kDynamicScale32QScratch = 0x0000000000000000ULL;
constexpr std::uint64_t kDynamicScale32KSrc1 = 0x0000000100062000ULL;
constexpr std::uint64_t kDynamicScale32KDst = 0x0000001000000e00ULL;
constexpr std::uint64_t kDynamicScale32KScale = 0x0000000200003800ULL;
constexpr std::uint64_t kDynamicScale32VSrc1 = 0x0000000100070000ULL;
constexpr std::uint64_t kDynamicScale32VDst = 0x0000001000000e80ULL;
constexpr std::uint64_t kDynamicScale32VScale = 0x0000000200004000ULL;
constexpr std::uint64_t kFusedQkvWeightBase = 0x0000000100000000ULL;
constexpr std::uint64_t kFusedQkvWeightLayerStride = 0x000000000071c000ULL;
constexpr std::uint64_t kFusedQkvMetadataBase = 0x0000000200000000ULL;
constexpr std::uint64_t kFusedQkvMetadataLayerStride = 0x0000000000031800ULL;
constexpr std::uint32_t kCommandRecordBytes = 68;
constexpr std::uint32_t kRopeRecordBytes = 128;
constexpr std::uint32_t kMaximumSequencePositions = 32768;
constexpr std::uint64_t kRopeTableBase = 0x0000000400000000ULL;
constexpr std::uint64_t kRmsnormBase = 0x0000000300000000ULL;
constexpr std::uint64_t kRmsnormGainBytes = 1792;
constexpr std::uint64_t kRmsnormScaleOffset = kRmsnormGainBytes + 8;
constexpr std::uint32_t kVocab = 151936;
constexpr std::uint32_t kHidden = 896;
constexpr std::uint32_t kTransformerLayers = 24;
constexpr std::uint32_t kLmTile = 32;
constexpr std::uint32_t kLastLmTile = kVocab / kLmTile - 1;
constexpr std::uint32_t kLmTileCount = kLastLmTile + 1;
constexpr std::uint32_t kEndOfTextToken = 151643;
constexpr std::uint32_t kImEndToken = 151645;
constexpr std::uint32_t kCsrControl = 0x18;
constexpr std::uint64_t kDefaultTimeoutCycles = 100000000ULL;
constexpr std::uint64_t kMinimumReadResponseLatencyCycles = 1ULL;

const std::array<const char*, 22> kOperatorNames = {
    "input_rmsnorm",
    "q_proj",
    "k_proj",
    "v_proj",
    "rope_q",
    "rope_k",
    "kv_write",
    "attention_score",
    "softmax",
    "attention_value",
    "attention_compose",
    "o_proj",
    "attention_residual_add",
    "post_attention_rmsnorm",
    "mlp_gate_proj",
    "mlp_up_proj",
    "silu_gate",
    "mlp_down_proj",
    "mlp_residual_add",
    "final_rmsnorm",
    "lm_head_tile",
    "fused_qkv",
};

struct BoundaryError : public std::runtime_error {
    std::string category;
    std::optional<std::uint64_t> address;

    BoundaryError(std::string category_in, std::string detail)
        : std::runtime_error(std::move(detail)), category(std::move(category_in)) {}

    BoundaryError(std::string category_in, std::string detail, std::uint64_t address_in)
        : std::runtime_error(std::move(detail)),
          category(std::move(category_in)),
          address(address_in) {}
};

std::string errno_string(const std::string& prefix) {
    return prefix + ": " + std::strerror(errno);
}

std::optional<std::uint64_t> read_json_unsigned_field(
    const fs::path& path,
    const std::string& key
) {
    if (!fs::exists(path)) {
        return std::nullopt;
    }
    std::ifstream input(path);
    const std::string raw(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>()
    );
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_offset = raw.find(needle);
    if (key_offset == std::string::npos) {
        return std::nullopt;
    }
    const std::size_t colon = raw.find(':', key_offset + needle.size());
    if (colon == std::string::npos) {
        return std::nullopt;
    }
    std::size_t begin = colon + 1;
    while (begin < raw.size() && std::isspace(static_cast<unsigned char>(raw[begin]))) {
        ++begin;
    }
    std::size_t end = begin;
    while (end < raw.size() && raw[end] >= '0' && raw[end] <= '9') {
        ++end;
    }
    if (end == begin) {
        return std::nullopt;
    }
    return std::stoull(raw.substr(begin, end - begin));
}

void write_all(int fd, const std::uint8_t* data, std::size_t size) {
    while (size != 0) {
        const ssize_t written = ::write(fd, data, size);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(errno_string("write failed"));
        }
        data += written;
        size -= static_cast<std::size_t>(written);
    }
}

void write_all(int fd, const std::string& text) {
    write_all(fd, reinterpret_cast<const std::uint8_t*>(text.data()), text.size());
}

void fsync_directory(const fs::path& path) {
    const fs::path directory = path.parent_path().empty() ? fs::path(".") : path.parent_path();
    const int fd = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd < 0) {
        throw std::runtime_error(errno_string("open directory for fsync failed"));
    }
    if (::fsync(fd) != 0) {
        const std::string message = errno_string("directory fsync failed");
        ::close(fd);
        throw std::runtime_error(message);
    }
    ::close(fd);
}

void write_atomic(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    const fs::path temporary = path.string() + ".tmp";
    const int fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error(errno_string("open temporary file failed"));
    }
    try {
        write_all(fd, text);
        if (::fsync(fd) != 0) {
            throw std::runtime_error(errno_string("file fsync failed"));
        }
        if (::close(fd) != 0) {
            throw std::runtime_error(errno_string("close temporary file failed"));
        }
    } catch (...) {
        ::close(fd);
        ::unlink(temporary.c_str());
        throw;
    }
    if (::rename(temporary.c_str(), path.c_str()) != 0) {
        ::unlink(temporary.c_str());
        throw std::runtime_error(errno_string("atomic rename failed"));
    }
    fsync_directory(path);
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned>(ch) << std::dec;
                } else {
                    out << ch;
                }
        }
    }
    return out.str();
}

std::string hex_digest(const std::array<std::uint8_t, SHA256_DIGEST_LENGTH>& digest) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (const std::uint8_t byte : digest) {
        out << std::setw(2) << static_cast<unsigned>(byte);
    }
    return out.str();
}

std::array<std::uint8_t, SHA256_DIGEST_LENGTH> sha256_bytes(
    const std::uint8_t* data,
    std::size_t size
) {
    std::array<std::uint8_t, SHA256_DIGEST_LENGTH> digest{};
    SHA256(data, size, digest.data());
    return digest;
}

std::uint64_t model_identity64(
    const std::array<std::uint8_t, SHA256_DIGEST_LENGTH>& model_sha
) {
    std::uint64_t identity = 0;
    for (std::size_t index = 0; index < 8; ++index) {
        identity |= static_cast<std::uint64_t>(model_sha[index]) << (index * 8);
    }
    return identity;
}

template <typename T>
T read_le(const std::vector<std::uint8_t>& data, std::size_t& offset) {
    static_assert(std::is_integral<T>::value, "read_le requires an integer type");
    if (offset + sizeof(T) > data.size()) {
        throw std::runtime_error("truncated little-endian value");
    }
    using U = typename std::make_unsigned<T>::type;
    U value = 0;
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        value |= static_cast<U>(data[offset + index]) << (index * 8);
    }
    offset += sizeof(T);
    return static_cast<T>(value);
}

template <typename T>
void append_le(std::vector<std::uint8_t>& data, T raw_value) {
    static_assert(std::is_integral<T>::value, "append_le requires an integer type");
    using U = typename std::make_unsigned<T>::type;
    const U value = static_cast<U>(raw_value);
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        data.push_back(static_cast<std::uint8_t>((value >> (index * 8)) & 0xffU));
    }
}

std::vector<std::uint8_t> read_file(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open file: " + path.string());
    }
    stream.seekg(0, std::ios::end);
    const auto end = stream.tellg();
    if (end < 0) {
        throw std::runtime_error("cannot size file: " + path.string());
    }
    std::vector<std::uint8_t> data(static_cast<std::size_t>(end));
    stream.seekg(0, std::ios::beg);
    if (!data.empty()) {
        stream.read(reinterpret_cast<char*>(data.data()), data.size());
    }
    if (!stream) {
        throw std::runtime_error("cannot read file: " + path.string());
    }
    return data;
}

struct Command {
    std::uint32_t ordinal = 0;
    std::uint16_t token_step = 0;
    std::uint8_t layer_id = 0;
    std::uint8_t operator_id = 0;
    std::uint8_t opcode = 0;
    std::uint8_t flags = 0;
    std::uint16_t m = 0;
    std::uint16_t n = 0;
    std::uint16_t k = 0;
    std::uint16_t sequence_position = 0;
    std::uint16_t completion_tag = 0;
    std::int16_t query_head = -1;
    std::int16_t context_token = -1;
    std::int32_t vocab_tile = -1;
    std::uint64_t src0 = 0;
    std::uint64_t src1 = 0;
    std::uint64_t dst = 0;
    std::uint64_t scale = 0;
    std::uint64_t scratch = 0;

    const char* operator_name() const {
        if (operator_id >= kOperatorNames.size()) {
            return "invalid_operator";
        }
        return kOperatorNames[operator_id];
    }
};

bool is_prompt_dynamic_scale32_rms(const Command& command, std::uint16_t position) {
    return command.token_step == position && command.layer_id == 0 &&
        command.operator_id == 0 && command.opcode == 2 &&
        command.flags == kDynamicScale32FrozenFlags && command.m == 1 &&
        command.n == kDynamicScale32InitialElements && command.k == 0 &&
        command.sequence_position == position &&
        command.query_head == -1 && command.context_token == -1 &&
        command.vocab_tile == -1 && command.src0 == kDynamicScale32RmsSrc0 &&
        command.src1 == kDynamicScale32RmsSrc1 && command.dst == kDynamicScale32RmsDst &&
        command.scale == kDynamicScale32RmsScale &&
        command.scratch == kDynamicScale32RmsScratch;
}

bool is_prompt_dynamic_scale32_projection(
    const Command& command,
    std::uint16_t position,
    std::uint8_t operator_id
) {
    const bool is_q = operator_id == 1;
    const bool is_k = operator_id == 2;
    const bool is_v = operator_id == 3;
    const std::uint16_t outputs = is_q ? 896U : 128U;
    const std::uint64_t src1 = is_q ? kDynamicScale32QSrc1 :
        (is_k ? kDynamicScale32KSrc1 : kDynamicScale32VSrc1);
    const std::uint64_t dst = is_q ? kDynamicScale32QDst :
        (is_k ? kDynamicScale32KDst : kDynamicScale32VDst);
    const std::uint64_t scale = is_q ? kDynamicScale32QScale :
        (is_k ? kDynamicScale32KScale : kDynamicScale32VScale);
    return (is_q || is_k || is_v) && command.token_step == position &&
        command.layer_id == 0 && command.operator_id == operator_id && command.opcode == 1 &&
        command.flags == kDynamicScale32FrozenFlags && command.m == 1 &&
        command.n == outputs && command.k == kDynamicScale32InitialElements &&
        command.sequence_position == position && command.query_head == -1 &&
        command.context_token == -1 && command.vocab_tile == -1 &&
        command.src0 == kDynamicScale32QSrc0 && command.src1 == src1 &&
        command.dst == dst && command.scale == scale &&
        command.scratch == kDynamicScale32QScratch;
}

struct DynamicScale32Sidecar {
    std::uint64_t payload_address = 0;
    std::array<std::uint8_t, 64> bytes{};
};

struct Package {
    std::uint32_t version = 0;
    std::uint32_t max_new_tokens = 0;
    std::uint32_t embedding_rows = 0;
    std::uint32_t embedding_cols = 0;
    std::uint64_t embedding_offset = 0;
    std::array<std::uint8_t, 32> schedule_sha{};
    std::array<std::uint8_t, 32> image_sha{};
    std::array<std::uint8_t, 32> model_sha{};
    std::array<std::uint8_t, 32> tokenizer_sha{};
    std::array<std::uint8_t, 32> prompt_tokens_sha{};
    std::array<std::uint8_t, 32> package_sha{};
    bool dynamic_scale32_enabled = false;
    std::uint32_t dynamic_scale32_feature_flags = 0;
    std::uint64_t dynamic_scale32_model_identity = 0;
    std::array<std::uint8_t, 32> dynamic_scale32_sidecars_sha{};
    std::vector<std::uint32_t> prompt_tokens;
    std::vector<std::uint32_t> termination_tokens;
    std::vector<std::uint8_t> rope_records;
    std::vector<DynamicScale32Sidecar> dynamic_scale32_sidecars;
    std::vector<Command> commands;
};

void read_digest(
    const std::vector<std::uint8_t>& data,
    std::size_t& offset,
    std::array<std::uint8_t, 32>& digest
) {
    if (offset + digest.size() > data.size()) {
        throw std::runtime_error("runtime package digest is truncated");
    }
    std::memcpy(digest.data(), data.data() + offset, digest.size());
    offset += digest.size();
}

void validate_dynamic_scale32_sidecar(
    const DynamicScale32Sidecar& record,
    std::uint64_t expected_model_identity
) {
    const auto& sidecar = record.bytes;
    if (record.payload_address < 64 || (record.payload_address & 63ULL) != 0) {
        throw std::runtime_error("ACE2RT2 dynamic Scale32 payload address is invalid");
    }
    if (std::memcmp(sidecar.data(), "BFP1", 4) != 0 || sidecar[4] != 1) {
        throw std::runtime_error("ACE2RT2 dynamic Scale32 sidecar magic or schema differs");
    }
    const std::uint32_t group_lanes = sidecar[5];
    const std::uint32_t group_count = sidecar[6];
    const std::uint32_t tensor_elements =
        static_cast<std::uint32_t>(sidecar[12]) |
        (static_cast<std::uint32_t>(sidecar[13]) << 8) |
        (static_cast<std::uint32_t>(sidecar[14]) << 16) |
        (static_cast<std::uint32_t>(sidecar[15]) << 24);
    const std::uint32_t expected_groups = group_lanes == 0
        ? 0
        : (tensor_elements + group_lanes - 1) / group_lanes;
    if ((group_lanes != 64 && group_lanes != 128) || group_count == 0 ||
        group_count > 38 || tensor_elements == 0 || expected_groups != group_count ||
        sidecar[7] != 0) {
        throw std::runtime_error("ACE2RT2 dynamic Scale32 sidecar shape differs");
    }
    std::uint64_t sidecar_model_identity = 0;
    for (std::size_t index = 0; index < 8; ++index) {
        sidecar_model_identity |= static_cast<std::uint64_t>(sidecar[16 + index]) << (index * 8);
    }
    if (sidecar_model_identity != expected_model_identity) {
        throw std::runtime_error("ACE2RT2 dynamic Scale32 sidecar model identity differs");
    }
    if (sidecar[62] != 0 || sidecar[63] != 0) {
        throw std::runtime_error("ACE2RT2 dynamic Scale32 sidecar reserved bytes differ");
    }
    for (std::size_t index = 0; index < 38; ++index) {
        const std::uint8_t raw_delta = sidecar[24 + index];
        if (index < group_count) {
            const std::int32_t delta = raw_delta & 0x80U
                ? static_cast<std::int32_t>(raw_delta) - 256
                : static_cast<std::int32_t>(raw_delta);
            if (delta < -24 || delta > 24) {
                throw std::runtime_error(
                    "ACE2RT2 dynamic Scale32 sidecar delta is outside [-24,24]"
                );
            }
        } else if (raw_delta != 0) {
            throw std::runtime_error("ACE2RT2 dynamic Scale32 sidecar reserved bytes differ");
        }
    }
}

void parse_dynamic_scale32_sidecars(
    const std::vector<std::uint8_t>& data,
    std::size_t& offset,
    std::uint32_t sidecar_count,
    Package& package
) {
    package.dynamic_scale32_sidecars.reserve(sidecar_count);
    for (std::uint32_t index = 0; index < sidecar_count; ++index) {
        DynamicScale32Sidecar record;
        record.payload_address = read_le<std::uint64_t>(data, offset);
        if (offset + record.bytes.size() > data.size()) {
            throw std::runtime_error("ACE2RT2 dynamic Scale32 sidecar array is truncated");
        }
        std::memcpy(record.bytes.data(), data.data() + offset, record.bytes.size());
        offset += record.bytes.size();
        const auto duplicate = std::find_if(
            package.dynamic_scale32_sidecars.begin(),
            package.dynamic_scale32_sidecars.end(),
            [&](const DynamicScale32Sidecar& existing) {
                return existing.payload_address == record.payload_address &&
                    existing.bytes[8] == record.bytes[8] &&
                    existing.bytes[9] == record.bytes[9] &&
                    existing.bytes[10] == record.bytes[10] &&
                    existing.bytes[11] == record.bytes[11];
            }
        );
        if (duplicate != package.dynamic_scale32_sidecars.end()) {
            throw std::runtime_error("ACE2RT2 dynamic Scale32 producer binding is duplicated");
        }
        validate_dynamic_scale32_sidecar(record, package.dynamic_scale32_model_identity);
        package.dynamic_scale32_sidecars.push_back(record);
    }
}

void parse_commands(
    const std::vector<std::uint8_t>& data,
    std::size_t& offset,
    std::uint32_t command_count,
    Package& package
) {
    if (command_count == 0 ||
        command_count > (data.size() - std::min(offset, data.size())) / kCommandRecordBytes) {
        throw std::runtime_error("runtime package command array is truncated or empty");
    }
    package.commands.reserve(command_count);
    for (std::uint32_t index = 0; index < command_count; ++index) {
        const std::size_t record_start = offset;
        Command command;
        command.ordinal = read_le<std::uint32_t>(data, offset);
        command.token_step = read_le<std::uint16_t>(data, offset);
        command.layer_id = read_le<std::uint8_t>(data, offset);
        command.operator_id = read_le<std::uint8_t>(data, offset);
        command.opcode = read_le<std::uint8_t>(data, offset);
        command.flags = read_le<std::uint8_t>(data, offset);
        command.m = read_le<std::uint16_t>(data, offset);
        command.n = read_le<std::uint16_t>(data, offset);
        command.k = read_le<std::uint16_t>(data, offset);
        command.sequence_position = read_le<std::uint16_t>(data, offset);
        command.completion_tag = read_le<std::uint16_t>(data, offset);
        command.query_head = read_le<std::int16_t>(data, offset);
        command.context_token = read_le<std::int16_t>(data, offset);
        command.vocab_tile = read_le<std::int32_t>(data, offset);
        command.src0 = read_le<std::uint64_t>(data, offset);
        command.src1 = read_le<std::uint64_t>(data, offset);
        command.dst = read_le<std::uint64_t>(data, offset);
        command.scale = read_le<std::uint64_t>(data, offset);
        command.scratch = read_le<std::uint64_t>(data, offset);
        if (offset - record_start != kCommandRecordBytes || command.ordinal != index ||
            command.operator_id >= kOperatorNames.size()) {
            throw std::runtime_error("runtime command package is not ordinally canonical");
        }
        package.commands.push_back(command);
    }
}

void validate_v2_command_steps(const Package& package) {
    std::optional<std::uint16_t> previous;
    std::uint16_t highest = 0;
    for (const Command& command : package.commands) {
        if (command.sequence_position != command.token_step) {
            throw std::runtime_error("ACE2RT2 sequence position differs from token step");
        }
        if (!previous.has_value()) {
            if (command.token_step != 0) {
                throw std::runtime_error("ACE2RT2 token steps do not begin at zero");
            }
        } else if (command.token_step < *previous || command.token_step > *previous + 1) {
            throw std::runtime_error("ACE2RT2 token steps are not contiguous and nondecreasing");
        }
        previous = command.token_step;
        highest = std::max(highest, command.token_step);
    }
    const std::uint64_t step_limit = static_cast<std::uint64_t>(package.prompt_tokens.size()) +
        package.max_new_tokens - 1;
    if (static_cast<std::uint64_t>(highest) + 1 != step_limit) {
        throw std::runtime_error("ACE2RT2 command stream does not cover the full generation bound");
    }

    std::size_t index = 0;
    auto take = [&](std::uint16_t step, std::uint8_t layer, std::uint8_t operator_id,
                    std::int16_t head, std::int16_t context, std::optional<std::uint8_t> flags,
                    std::optional<std::int32_t> vocab_tile) -> const Command& {
        if (index >= package.commands.size()) {
            throw std::runtime_error("ACE2RT2 full schedule is truncated");
        }
        const Command& command = package.commands[index];
        const std::array<std::uint8_t, 22> expected_opcodes = {
            2, 1, 1, 1, 3, 3, 10, 4, 5, 6, 9, 1, 8, 2, 1, 1, 7, 1, 8, 2, 1, 11
        };
        if (command.ordinal != index || command.completion_tag != (index & 0xffffU) ||
            command.token_step != step || command.sequence_position != step ||
            command.layer_id != layer || command.operator_id != operator_id ||
            command.opcode != expected_opcodes.at(operator_id) || command.query_head != head ||
            command.context_token != context ||
            (flags.has_value() && command.flags != *flags) ||
            (vocab_tile.has_value() && command.vocab_tile != *vocab_tile) ||
            (!vocab_tile.has_value() && command.vocab_tile != -1)) {
            throw std::runtime_error("ACE2RT2 full operator schedule differs at command " +
                                     std::to_string(index));
        }
        ++index;
        return command;
    };
    const std::array<std::uint8_t, 7> prefix = {0, 1, 2, 3, 4, 5, 6};
    const std::array<std::uint8_t, 8> suffix = {11, 12, 13, 14, 15, 16, 17, 18};
    bool saw_fused_qkv = false;
    bool saw_ordinary_legacy_qkv = false;
    for (std::uint32_t step = 0; step < step_limit; ++step) {
        for (std::uint8_t layer = 0; layer < 24; ++layer) {
            take(static_cast<std::uint16_t>(step), layer, prefix[0], -1, -1,
                 std::nullopt, std::nullopt);
            if (index < package.commands.size() && package.commands[index].operator_id == 21) {
                const Command& fused = take(
                    static_cast<std::uint16_t>(step), layer, 21, -1, -1,
                    std::uint8_t{0}, std::nullopt
                );
                const auto range_within = [](std::uint64_t address, std::uint64_t bytes,
                                             std::uint64_t base, std::uint64_t region_bytes) {
                    return address >= base && bytes <= region_bytes &&
                        address - base <= region_bytes - bytes;
                };
                const std::uint64_t expected_weight =
                    kFusedQkvWeightBase +
                    static_cast<std::uint64_t>(layer) * kFusedQkvWeightLayerStride;
                const std::uint64_t expected_metadata =
                    kFusedQkvMetadataBase +
                    static_cast<std::uint64_t>(layer) * kFusedQkvMetadataLayerStride;
                if (fused.m != 1 || fused.n != 896 || fused.k != 896 ||
                    fused.scratch != 0 || fused.src0 != kDynamicScale32QSrc0 ||
                    fused.dst != kDynamicScale32QDst ||
                    fused.src1 != expected_weight || fused.scale != expected_metadata ||
                    (fused.src1 & 15ULL) != 0 || (fused.scale & 15ULL) != 0 ||
                    !range_within(fused.src1, 516096ULL,
                                  0x0000000100000000ULL, 246980608ULL) ||
                    !range_within(fused.scale, 18432ULL,
                                  0x0000000200000000ULL, 7297024ULL)) {
                    throw std::runtime_error(
                        "ACE2RT2 fused-QKV descriptor geometry or address differs"
                    );
                }
                saw_fused_qkv = true;
            } else {
                const Command& q = take(
                    static_cast<std::uint16_t>(step), layer, prefix[1], -1, -1,
                    std::nullopt, std::nullopt
                );
                take(static_cast<std::uint16_t>(step), layer, prefix[2], -1, -1,
                     std::nullopt, std::nullopt);
                take(static_cast<std::uint16_t>(step), layer, prefix[3], -1, -1,
                     std::nullopt, std::nullopt);
                if ((q.flags & kDynamicScale32Flag) == 0) {
                    saw_ordinary_legacy_qkv = true;
                }
            }
            for (const std::uint8_t operator_id : {prefix[4], prefix[5], prefix[6]}) {
                take(static_cast<std::uint16_t>(step), layer, operator_id, -1, -1,
                     std::nullopt, std::nullopt);
            }
            if (step == 0) {
                for (std::int16_t head = 0; head < 14; ++head) {
                    take(0, layer, 7, head, 0, std::nullopt, std::nullopt);
                    take(0, layer, 8, head, -1, std::nullopt, std::nullopt);
                    take(0, layer, 9, head, 0, std::nullopt, std::nullopt);
                }
            } else {
                const std::uint32_t context_count = step + 1;
                for (std::int16_t head = 0; head < 14; ++head) {
                    for (std::uint32_t context = 0; context < context_count; ++context) {
                        take(static_cast<std::uint16_t>(step), layer, 7, head,
                             static_cast<std::int16_t>(context), std::nullopt, std::nullopt);
                    }
                    for (const auto phases : {std::pair<std::uint8_t, std::uint8_t>{0, 1},
                                              std::pair<std::uint8_t, std::uint8_t>{2, 3}}) {
                        for (std::uint32_t context = 0; context < context_count; ++context) {
                            take(static_cast<std::uint16_t>(step), layer, 10, head,
                                 static_cast<std::int16_t>(context),
                                 context == 0 ? phases.first : phases.second, std::nullopt);
                        }
                    }
                    for (std::uint32_t context = 0; context < context_count; ++context) {
                        const std::uint8_t phase = context == 0 ? 4 :
                            (context + 1 == context_count ? 6 : 5);
                        take(static_cast<std::uint16_t>(step), layer, 10, head,
                             static_cast<std::int16_t>(context), phase, std::nullopt);
                    }
                }
            }
            for (const std::uint8_t operator_id : suffix) {
                take(static_cast<std::uint16_t>(step), layer, operator_id, -1, -1,
                     std::nullopt, std::nullopt);
            }
        }
        take(static_cast<std::uint16_t>(step), 24, 19, -1, -1,
             std::nullopt, std::nullopt);
        for (std::int32_t tile = 0; tile <= static_cast<std::int32_t>(kLastLmTile); ++tile) {
            take(static_cast<std::uint16_t>(step), 24, 20, -1, -1,
                 std::nullopt, tile);
        }
    }
    if (index != package.commands.size()) {
        throw std::runtime_error("ACE2RT2 full schedule has trailing commands");
    }
    if (saw_fused_qkv && saw_ordinary_legacy_qkv) {
        throw std::runtime_error("ACE2RT2 QKV schedule mixes fused and ordinary legacy groups");
    }
}

void validate_dynamic_scale32_initial_binding(const Package& package) {
    const auto dynamic_count = std::count_if(
        package.commands.begin(), package.commands.end(),
        [](const Command& command) {
            return (command.flags & kDynamicScale32Flag) != 0;
        }
    );
    if (!package.dynamic_scale32_enabled) {
        if (dynamic_count != 0) {
            throw std::runtime_error(
                "ACE2RT2 flags[6] initial boundary lacks its DS32 sidecar"
            );
        }
        return;
    }
    const std::size_t prompt_positions = package.prompt_tokens.size();
    if (package.dynamic_scale32_sidecars.size() != prompt_positions ||
        dynamic_count != prompt_positions * 4U) {
        throw std::runtime_error(
            "ACE2RT2 prompt DS32 tranche requires one host plan and RMS/Q/K/V per prompt position"
        );
    }

    for (const Command& command : package.commands) {
        const bool expected = command.token_step < prompt_positions &&
            command.layer_id == 0 && command.operator_id <= 3;
        if (((command.flags & kDynamicScale32Flag) != 0) != expected) {
            throw std::runtime_error(
                "ACE2RT2 flags[6] is only legal on prompt-position layer-0 RMS/Q/K/V"
            );
        }
    }

    for (std::size_t position = 0; position < prompt_positions; ++position) {
        std::array<const Command*, 4> tranche{{nullptr, nullptr, nullptr, nullptr}};
        for (const Command& command : package.commands) {
            if (command.token_step == position && command.layer_id == 0 &&
                command.operator_id <= 3) {
                tranche.at(command.operator_id) = &command;
            }
        }
        if (std::any_of(tranche.begin(), tranche.end(), [](const Command* command) {
                return command == nullptr;
            })) {
            throw std::runtime_error("ACE2RT2 prompt DS32 operator order differs");
        }
        if (!is_prompt_dynamic_scale32_rms(*tranche[0], static_cast<std::uint16_t>(position)) ||
            !is_prompt_dynamic_scale32_projection(
                *tranche[1], static_cast<std::uint16_t>(position), 1
            ) ||
            !is_prompt_dynamic_scale32_projection(
                *tranche[2], static_cast<std::uint16_t>(position), 2
            ) ||
            !is_prompt_dynamic_scale32_projection(
                *tranche[3], static_cast<std::uint16_t>(position), 3
            )) {
            throw std::runtime_error("ACE2RT2 prompt DS32 command contract differs");
        }

        const DynamicScale32Sidecar& record = package.dynamic_scale32_sidecars.at(position);
        const auto& sidecar = record.bytes;
        const std::uint16_t producer_tag = static_cast<std::uint16_t>(sidecar[8]) |
            (static_cast<std::uint16_t>(sidecar[9]) << 8);
        const std::uint32_t elements = static_cast<std::uint32_t>(sidecar[12]) |
            (static_cast<std::uint32_t>(sidecar[13]) << 8) |
            (static_cast<std::uint32_t>(sidecar[14]) << 16) |
            (static_cast<std::uint32_t>(sidecar[15]) << 24);
        if (record.payload_address != tranche[0]->src0 ||
            sidecar[5] != kDynamicScale32InitialGroupLanes ||
            sidecar[6] != kDynamicScale32InitialGroupCount ||
            producer_tag != position ||
            sidecar[10] != kDynamicScale32HostPlanLayer ||
            sidecar[11] != kDynamicScale32HostPlanOpcode ||
            elements != kDynamicScale32InitialElements) {
            throw std::runtime_error("ACE2RT2 prompt DS32 host-plan producer contract differs");
        }
    }
}

Package load_package(
    const fs::path& path,
    const ace2_runtime_identity::IdentityProfile& identity_profile
) {
    const std::vector<std::uint8_t> data = read_file(path);
    if (data.size() < 8) {
        throw std::runtime_error("runtime package magic differs");
    }
    Package package;
    package.package_sha = sha256_bytes(data.data(), data.size());
    std::size_t offset = 8;
    if (std::memcmp(data.data(), "ACE2RT1\0", 8) == 0) {
        package.version = read_le<std::uint32_t>(data, offset);
        const std::uint32_t record_bytes = read_le<std::uint32_t>(data, offset);
        const std::uint32_t command_count = read_le<std::uint32_t>(data, offset);
        const std::uint32_t seed_token = read_le<std::uint32_t>(data, offset);
        package.embedding_rows = read_le<std::uint32_t>(data, offset);
        package.embedding_cols = read_le<std::uint32_t>(data, offset);
        package.embedding_offset = read_le<std::uint64_t>(data, offset);
        if (package.version != kPackageVersion1 || record_bytes != kCommandRecordBytes) {
            throw std::runtime_error("runtime package version or record width differs");
        }
        read_digest(data, offset, package.schedule_sha);
        read_digest(data, offset, package.image_sha);
        read_digest(data, offset, package.model_sha);
        constexpr std::size_t kRopeBytes = 256;
        if (offset + kRopeBytes > data.size()) {
            throw std::runtime_error("runtime package RoPE records are truncated");
        }
        package.rope_records.assign(data.begin() + static_cast<std::ptrdiff_t>(offset),
                                    data.begin() + static_cast<std::ptrdiff_t>(offset + kRopeBytes));
        offset += kRopeBytes;
        package.prompt_tokens.push_back(seed_token);
        package.max_new_tokens = 2;
        parse_commands(data, offset, command_count, package);
        if (offset != data.size()) {
            throw std::runtime_error("runtime package has trailing bytes");
        }
        if (package.commands.size() != 13914 || package.embedding_rows != kVocab ||
            package.embedding_cols != kHidden || seed_token >= kVocab) {
            throw std::runtime_error("runtime package geometry differs from the accepted contract");
        }
        if (identity_profile.require_exact_package_sha256) {
            throw std::runtime_error("ACE2RT2 pinned model or tokenizer identity differs");
        }
        return package;
    }
    if (std::memcmp(data.data(), "ACE2RT2\0", 8) != 0) {
        throw std::runtime_error("runtime package magic differs");
    }

    package.version = read_le<std::uint32_t>(data, offset);
    const std::uint32_t header_bytes = read_le<std::uint32_t>(data, offset);
    const std::uint32_t record_bytes = read_le<std::uint32_t>(data, offset);
    const std::uint32_t command_count = read_le<std::uint32_t>(data, offset);
    const std::uint32_t prompt_token_count = read_le<std::uint32_t>(data, offset);
    package.max_new_tokens = read_le<std::uint32_t>(data, offset);
    const std::uint32_t termination_token_count = read_le<std::uint32_t>(data, offset);
    package.embedding_rows = read_le<std::uint32_t>(data, offset);
    package.embedding_cols = read_le<std::uint32_t>(data, offset);
    const std::uint32_t rope_position_count = read_le<std::uint32_t>(data, offset);
    package.embedding_offset = read_le<std::uint64_t>(data, offset);
    if (package.version != kPackageVersion2 ||
        (header_bytes != kPackageV2HeaderBytes &&
         header_bytes != kPackageV2HeaderBytes + kPackageV2DynamicExtensionBytes) ||
        record_bytes != kCommandRecordBytes || prompt_token_count == 0 ||
        package.max_new_tokens == 0 || package.max_new_tokens > 32 ||
        termination_token_count != 2 || package.embedding_rows != kVocab ||
        package.embedding_cols != kHidden || rope_position_count == 0 ||
        rope_position_count > kMaximumSequencePositions ||
        static_cast<std::uint64_t>(rope_position_count) !=
            static_cast<std::uint64_t>(prompt_token_count) + package.max_new_tokens - 1) {
        throw std::runtime_error("ACE2RT2 header contract differs");
    }
    read_digest(data, offset, package.schedule_sha);
    read_digest(data, offset, package.image_sha);
    read_digest(data, offset, package.model_sha);
    read_digest(data, offset, package.tokenizer_sha);
    read_digest(data, offset, package.prompt_tokens_sha);
    std::uint32_t dynamic_scale32_sidecar_count = 0;
    if (header_bytes != kPackageV2HeaderBytes) {
        if (offset + 4 > data.size() || std::memcmp(data.data() + offset, "DS32", 4) != 0) {
            throw std::runtime_error("ACE2RT2 dynamic Scale32 extension contract differs");
        }
        offset += 4;
        const std::uint32_t extension_version = read_le<std::uint32_t>(data, offset);
        const std::uint32_t sidecar_record_bytes = read_le<std::uint32_t>(data, offset);
        dynamic_scale32_sidecar_count = read_le<std::uint32_t>(data, offset);
        package.dynamic_scale32_model_identity = read_le<std::uint64_t>(data, offset);
        read_digest(data, offset, package.dynamic_scale32_sidecars_sha);
        package.dynamic_scale32_feature_flags = read_le<std::uint32_t>(data, offset);
        const std::uint32_t extension_reserved = read_le<std::uint32_t>(data, offset);
        if (extension_version != 1 || sidecar_record_bytes != kPackageV2DynamicRecordBytes ||
            dynamic_scale32_sidecar_count == 0 ||
            package.dynamic_scale32_model_identity != model_identity64(package.model_sha) ||
            package.dynamic_scale32_feature_flags != kDynamicScale32FeatureSidecars ||
            extension_reserved != 0) {
            throw std::runtime_error("ACE2RT2 dynamic Scale32 extension contract differs");
        }
        package.dynamic_scale32_enabled = true;
    }
    if (!ace2_runtime_identity::accepts_identity(
            identity_profile,
            hex_digest(package.package_sha),
            hex_digest(package.image_sha),
            hex_digest(package.model_sha),
            hex_digest(package.tokenizer_sha),
            package.embedding_offset
        )) {
        throw std::runtime_error("ACE2RT2 pinned model or tokenizer identity differs");
    }
    if (offset != header_bytes ||
        prompt_token_count > (data.size() - std::min(offset, data.size())) / 4) {
        throw std::runtime_error("ACE2RT2 prompt token array is truncated");
    }
    const std::size_t prompt_start = offset;
    package.prompt_tokens.reserve(prompt_token_count);
    for (std::uint32_t index = 0; index < prompt_token_count; ++index) {
        const std::uint32_t token = read_le<std::uint32_t>(data, offset);
        if (token >= package.embedding_rows) {
            throw std::runtime_error("ACE2RT2 prompt token is outside the embedding table");
        }
        package.prompt_tokens.push_back(token);
    }
    const auto prompt_digest = sha256_bytes(
        data.data() + prompt_start, offset - prompt_start
    );
    if (prompt_digest != package.prompt_tokens_sha) {
        throw std::runtime_error("ACE2RT2 prompt-token digest differs");
    }
    package.termination_tokens.reserve(termination_token_count);
    for (std::uint32_t index = 0; index < termination_token_count; ++index) {
        package.termination_tokens.push_back(read_le<std::uint32_t>(data, offset));
    }
    if (package.termination_tokens != std::vector<std::uint32_t>{kEndOfTextToken, kImEndToken}) {
        throw std::runtime_error("ACE2RT2 termination token list differs");
    }
    const std::size_t manifest_start = offset;
    const std::uint64_t rope_bytes = static_cast<std::uint64_t>(rope_position_count) *
        kRopeRecordBytes;
    if (rope_bytes > data.size() - std::min(offset, data.size())) {
        throw std::runtime_error("ACE2RT2 RoPE record array is truncated");
    }
    package.rope_records.assign(
        data.begin() + static_cast<std::ptrdiff_t>(offset),
        data.begin() + static_cast<std::ptrdiff_t>(offset + rope_bytes)
    );
    offset += static_cast<std::size_t>(rope_bytes);
    if (dynamic_scale32_sidecar_count >
        (data.size() - std::min(offset, data.size())) / kPackageV2DynamicRecordBytes) {
        throw std::runtime_error("ACE2RT2 dynamic Scale32 sidecar array is truncated");
    }
    const std::size_t sidecar_start = offset;
    const std::size_t sidecar_bytes =
        static_cast<std::size_t>(dynamic_scale32_sidecar_count) *
        kPackageV2DynamicRecordBytes;
    const auto sidecar_digest = sha256_bytes(data.data() + sidecar_start, sidecar_bytes);
    if (package.dynamic_scale32_enabled &&
        sidecar_digest != package.dynamic_scale32_sidecars_sha) {
        throw std::runtime_error("ACE2RT2 dynamic Scale32 sidecar digest differs");
    }
    if (!package.dynamic_scale32_enabled) {
        package.dynamic_scale32_model_identity = model_identity64(package.model_sha);
        package.dynamic_scale32_sidecars_sha = sidecar_digest;
    }
    parse_dynamic_scale32_sidecars(
        data, offset, dynamic_scale32_sidecar_count, package
    );
    const std::size_t command_start = offset;
    parse_commands(data, offset, command_count, package);
    if (offset != data.size()) {
        throw std::runtime_error("runtime package has trailing bytes");
    }
    const auto schedule_digest = sha256_bytes(
        data.data() + manifest_start, data.size() - manifest_start
    );
    if (schedule_digest != package.schedule_sha) {
        throw std::runtime_error("ACE2RT2 schedule digest differs");
    }
    validate_v2_command_steps(package);
    validate_dynamic_scale32_initial_binding(package);
    return package;
}

struct Beat {
    std::array<std::uint8_t, 16> data{};
    std::uint16_t valid = 0;
};

struct ImageRegion {
    std::uint64_t base;
    std::uint64_t bytes;
    std::uint64_t file_offset;
};

class Memory {
  public:
    Memory(
        const fs::path& image_path,
        const ace2_runtime_identity::IdentityProfile& identity_profile
    ) {
        image_fd_ = ::open(image_path.c_str(), O_RDONLY);
        if (image_fd_ < 0) {
            throw std::runtime_error(errno_string("open sealed image failed"));
        }
        struct stat metadata {};
        if (::fstat(image_fd_, &metadata) != 0) {
            throw std::runtime_error(errno_string("stat sealed image failed"));
        }
        image_bytes_ = static_cast<std::size_t>(metadata.st_size);
        if (image_bytes_ != identity_profile.image_bytes) {
            throw std::runtime_error("sealed image byte count differs");
        }
        if (identity_profile.image_layout ==
            ace2_runtime_identity::ImageLayoutKind::SyntheticR6FullScale32) {
            regions_ = {{
                {0x0000000100000000ULL, 246980608ULL, 0ULL},
                {0x0000000200000000ULL, 61745152ULL, 246980608ULL},
                {0x0000000300000000ULL, 88592ULL, 308725760ULL},
                {0x0000000310000000ULL, 55296ULL, 308814352ULL},
            }};
        } else {
            regions_ = {{
                {0x0000000100000000ULL, 246980608ULL, 0ULL},
                {0x0000000200000000ULL, 7297024ULL, 246980608ULL},
                {0x0000000300000000ULL, 88592ULL, 254277632ULL},
                {0x0000000310000000ULL, 55296ULL, 254366224ULL},
            }};
        }
        void* mapped = ::mmap(
            nullptr, image_bytes_, PROT_READ, MAP_PRIVATE, image_fd_, 0
        );
        if (mapped == MAP_FAILED) {
            throw std::runtime_error(errno_string("mmap sealed image failed"));
        }
        image_ = static_cast<const std::uint8_t*>(mapped);
    }

    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;

    ~Memory() {
        if (image_ != nullptr) {
            ::munmap(const_cast<std::uint8_t*>(image_), image_bytes_);
        }
        if (image_fd_ >= 0) {
            ::close(image_fd_);
        }
    }

    bool in_image(std::uint64_t address, std::size_t bytes, std::size_t& file_offset) const {
        for (const auto& region : regions_) {
            if (address >= region.base &&
                address - region.base <= region.bytes &&
                bytes <= region.bytes - (address - region.base)) {
                file_offset = static_cast<std::size_t>(
                    region.file_offset + address - region.base
                );
                return true;
            }
        }
        return false;
    }

    std::array<std::uint8_t, 16> read_beat(std::uint64_t address) const {
        if ((address & 15ULL) != 0) {
            throw BoundaryError("memory_alignment", "unaligned 16-byte read", address);
        }
        std::size_t file_offset = 0;
        std::array<std::uint8_t, 16> result{};
        if (in_image(address, result.size(), file_offset)) {
            std::memcpy(result.data(), image_ + file_offset, result.size());
            return result;
        }
        const auto found = mutable_.find(address);
        if (found == mutable_.end() || found->second.valid != 0xffffU) {
            throw BoundaryError(
                "missing_memory",
                "read reached an uninitialized runtime address",
                address
            );
        }
        return found->second.data;
    }

    std::vector<std::uint8_t> read_range(std::uint64_t address, std::size_t bytes) const {
        std::vector<std::uint8_t> result(bytes);
        for (std::size_t index = 0; index < bytes; ++index) {
            const std::uint64_t beat_address = (address + index) & ~15ULL;
            const auto beat = read_beat(beat_address);
            result[index] = beat[(address + index) & 15ULL];
        }
        return result;
    }

    void write_beat(
        std::uint64_t address,
        const std::array<std::uint8_t, 16>& data,
        std::uint16_t strobe
    ) {
        if ((address & 15ULL) != 0) {
            throw BoundaryError("memory_alignment", "unaligned 16-byte write", address);
        }
        std::size_t ignored = 0;
        if (in_image(address, 16, ignored)) {
            throw BoundaryError("memory_protection", "write targeted sealed image data", address);
        }
        Beat& beat = mutable_[address];
        for (std::size_t lane = 0; lane < 16; ++lane) {
            if ((strobe >> lane) & 1U) {
                beat.data[lane] = data[lane];
                beat.valid |= static_cast<std::uint16_t>(1U << lane);
            }
        }
    }

    void preload(std::uint64_t address, const std::uint8_t* data, std::size_t bytes) {
        for (std::size_t index = 0; index < bytes; ++index) {
            const std::uint64_t absolute = address + index;
            const std::uint64_t beat_address = absolute & ~15ULL;
            Beat& beat = mutable_[beat_address];
            const unsigned lane = static_cast<unsigned>(absolute & 15ULL);
            beat.data[lane] = data[index];
            beat.valid |= static_cast<std::uint16_t>(1U << lane);
        }
    }

  private:
    int image_fd_ = -1;
    const std::uint8_t* image_ = nullptr;
    std::size_t image_bytes_ = 0;
    std::array<ImageRegion, 4> regions_{};
    std::unordered_map<std::uint64_t, Beat> mutable_;
};

void stage_dynamic_scale32_sidecar_for_step(
    Memory& memory,
    const Package& package,
    std::uint32_t token_step
) {
    if (!package.dynamic_scale32_enabled || token_step >= package.prompt_tokens.size()) {
        return;
    }
    const DynamicScale32Sidecar& record = package.dynamic_scale32_sidecars.at(token_step);
    const std::uint64_t sidecar_address = record.payload_address - record.bytes.size();
    std::size_t ignored = 0;
    if (memory.in_image(sidecar_address, record.bytes.size(), ignored)) {
        throw BoundaryError(
            "memory_protection",
            "dynamic Scale32 sidecar targeted sealed image data",
            sidecar_address
        );
    }
    memory.preload(sidecar_address, record.bytes.data(), record.bytes.size());
}

class EmbeddingSource {
  public:
    EmbeddingSource(const fs::path& path, std::uint64_t data_offset)
        : data_offset_(data_offset) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error(errno_string("open raw safetensors failed"));
        }
    }

    EmbeddingSource(const EmbeddingSource&) = delete;
    EmbeddingSource& operator=(const EmbeddingSource&) = delete;

    ~EmbeddingSource() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    std::vector<std::uint8_t> quantized_row(std::uint32_t token, float scale) const {
        if (token >= kVocab || !std::isfinite(scale) || scale <= 0.0f) {
            throw BoundaryError("embedding_preload", "invalid embedding token or scale");
        }
        std::array<std::uint8_t, kHidden * 2> raw{};
        const off_t offset = static_cast<off_t>(
            data_offset_ + static_cast<std::uint64_t>(token) * raw.size()
        );
        std::size_t consumed = 0;
        while (consumed < raw.size()) {
            const ssize_t count = ::pread(
                fd_, raw.data() + consumed, raw.size() - consumed,
                offset + static_cast<off_t>(consumed)
            );
            if (count < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error(errno_string("read embedding row failed"));
            }
            if (count == 0) {
                throw std::runtime_error("embedding row is truncated");
            }
            consumed += static_cast<std::size_t>(count);
        }
        std::vector<std::uint8_t> output(kHidden);
        for (std::size_t lane = 0; lane < kHidden; ++lane) {
            const std::uint16_t bf16 = static_cast<std::uint16_t>(raw[lane * 2]) |
                (static_cast<std::uint16_t>(raw[lane * 2 + 1]) << 8);
            const std::uint32_t fp32_bits = static_cast<std::uint32_t>(bf16) << 16;
            float value = 0.0f;
            std::memcpy(&value, &fp32_bits, sizeof(value));
            float rounded = std::nearbyintf(value / scale);
            rounded = std::max(-128.0f, std::min(127.0f, rounded));
            output[lane] = static_cast<std::uint8_t>(
                static_cast<std::int8_t>(static_cast<int>(rounded))
            );
        }
        return output;
    }

  private:
    int fd_ = -1;
    std::uint64_t data_offset_ = 0;
};

struct WriteRecord {
    std::uint64_t address = 0;
    std::uint16_t strobe = 0;
    std::array<std::uint8_t, 16> data{};
};

struct CompletedRecord {
    std::uint32_t ordinal = 0;
    std::uint64_t cycles = 0;
    std::uint32_t read_beats = 0;
    std::uint32_t write_beats = 0;
    std::uint16_t done_tag = 0;
    bool done_error = false;
    bool saturation = false;
    std::int32_t argmax_token = -1;
    std::int32_t argmax_logit = -129;
    std::vector<std::uint32_t> generated_tokens;
    bool terminated = false;
    std::array<std::uint8_t, 32> source_sha{};
    std::array<std::uint8_t, 32> destination_sha{};
    std::vector<WriteRecord> writes;
};

std::vector<std::uint8_t> serialize_completed(const CompletedRecord& record) {
    std::vector<std::uint8_t> body;
    append_le(body, record.ordinal);
    append_le(body, record.cycles);
    append_le(body, record.read_beats);
    append_le(body, record.write_beats);
    append_le(body, record.done_tag);
    append_le(body, static_cast<std::uint8_t>(record.done_error));
    append_le(body, static_cast<std::uint8_t>(record.saturation));
    append_le(body, record.argmax_token);
    append_le(body, record.argmax_logit);
    append_le(body, static_cast<std::uint32_t>(record.generated_tokens.size()));
    for (const std::uint32_t token : record.generated_tokens) {
        append_le(body, token);
    }
    append_le(body, static_cast<std::uint8_t>(record.terminated));
    body.insert(body.end(), record.source_sha.begin(), record.source_sha.end());
    body.insert(body.end(), record.destination_sha.begin(), record.destination_sha.end());
    append_le(body, static_cast<std::uint32_t>(record.writes.size()));
    for (const auto& write : record.writes) {
        append_le(body, write.address);
        append_le(body, write.strobe);
        body.insert(body.end(), write.data.begin(), write.data.end());
    }
    return body;
}

std::vector<std::uint8_t> serialize_completed_v1(const CompletedRecord& record) {
    if (record.generated_tokens.size() > 2 || record.terminated) {
        throw std::runtime_error("legacy journal cannot encode generalized token state");
    }
    std::vector<std::uint8_t> body;
    append_le(body, record.ordinal);
    append_le(body, record.cycles);
    append_le(body, record.read_beats);
    append_le(body, record.write_beats);
    append_le(body, record.done_tag);
    append_le(body, static_cast<std::uint8_t>(record.done_error));
    append_le(body, static_cast<std::uint8_t>(record.saturation));
    append_le(body, record.argmax_token);
    append_le(body, record.argmax_logit);
    append_le(body, record.generated_tokens.empty()
        ? std::int32_t{-1} : static_cast<std::int32_t>(record.generated_tokens[0]));
    append_le(body, record.generated_tokens.size() < 2
        ? std::int32_t{-1} : static_cast<std::int32_t>(record.generated_tokens[1]));
    body.insert(body.end(), record.source_sha.begin(), record.source_sha.end());
    body.insert(body.end(), record.destination_sha.begin(), record.destination_sha.end());
    append_le(body, static_cast<std::uint32_t>(record.writes.size()));
    for (const auto& write : record.writes) {
        append_le(body, write.address);
        append_le(body, write.strobe);
        body.insert(body.end(), write.data.begin(), write.data.end());
    }
    return body;
}

void parse_completed_writes(
    const std::vector<std::uint8_t>& body,
    std::size_t& offset,
    CompletedRecord& record
) {
    for (auto* digest : {&record.source_sha, &record.destination_sha}) {
        if (offset + digest->size() > body.size()) {
            throw std::runtime_error("journal digest is truncated");
        }
        std::memcpy(digest->data(), body.data() + offset, digest->size());
        offset += digest->size();
    }
    const std::uint32_t write_count = read_le<std::uint32_t>(body, offset);
    record.writes.reserve(write_count);
    for (std::uint32_t index = 0; index < write_count; ++index) {
        WriteRecord write;
        write.address = read_le<std::uint64_t>(body, offset);
        write.strobe = read_le<std::uint16_t>(body, offset);
        if (offset + write.data.size() > body.size()) {
            throw std::runtime_error("journal write data is truncated");
        }
        std::memcpy(write.data.data(), body.data() + offset, write.data.size());
        offset += write.data.size();
        record.writes.push_back(write);
    }
    if (offset != body.size() || record.write_beats != record.writes.size()) {
        throw std::runtime_error("journal frame width differs");
    }
}

CompletedRecord parse_completed(const std::vector<std::uint8_t>& body) {
    std::size_t offset = 0;
    CompletedRecord record;
    record.ordinal = read_le<std::uint32_t>(body, offset);
    record.cycles = read_le<std::uint64_t>(body, offset);
    record.read_beats = read_le<std::uint32_t>(body, offset);
    record.write_beats = read_le<std::uint32_t>(body, offset);
    record.done_tag = read_le<std::uint16_t>(body, offset);
    record.done_error = read_le<std::uint8_t>(body, offset) != 0;
    record.saturation = read_le<std::uint8_t>(body, offset) != 0;
    record.argmax_token = read_le<std::int32_t>(body, offset);
    record.argmax_logit = read_le<std::int32_t>(body, offset);
    const std::uint32_t generated_count = read_le<std::uint32_t>(body, offset);
    if (generated_count > 32) {
        throw std::runtime_error("journal generated-token count exceeds the runtime bound");
    }
    record.generated_tokens.reserve(generated_count);
    for (std::uint32_t index = 0; index < generated_count; ++index) {
        record.generated_tokens.push_back(read_le<std::uint32_t>(body, offset));
    }
    record.terminated = read_le<std::uint8_t>(body, offset) != 0;
    parse_completed_writes(body, offset, record);
    return record;
}

CompletedRecord parse_completed_v1(const std::vector<std::uint8_t>& body) {
    std::size_t offset = 0;
    CompletedRecord record;
    record.ordinal = read_le<std::uint32_t>(body, offset);
    record.cycles = read_le<std::uint64_t>(body, offset);
    record.read_beats = read_le<std::uint32_t>(body, offset);
    record.write_beats = read_le<std::uint32_t>(body, offset);
    record.done_tag = read_le<std::uint16_t>(body, offset);
    record.done_error = read_le<std::uint8_t>(body, offset) != 0;
    record.saturation = read_le<std::uint8_t>(body, offset) != 0;
    record.argmax_token = read_le<std::int32_t>(body, offset);
    record.argmax_logit = read_le<std::int32_t>(body, offset);
    const std::int32_t generated0 = read_le<std::int32_t>(body, offset);
    const std::int32_t generated1 = read_le<std::int32_t>(body, offset);
    if (generated0 >= 0) record.generated_tokens.push_back(generated0);
    if (generated1 >= 0) record.generated_tokens.push_back(generated1);
    parse_completed_writes(body, offset, record);
    return record;
}

class Journal {
  public:
    Journal(const fs::path& path, const Package& package, bool resume)
        : path_(path), package_(package) {
        if (resume) {
            records_ = load_existing();
        } else {
            if (fs::exists(path_)) {
                throw std::runtime_error("fresh runtime journal already exists");
            }
            create_header();
        }
        fd_ = ::open(path_.c_str(), O_WRONLY | O_APPEND);
        if (fd_ < 0) {
            throw std::runtime_error(errno_string("open runtime journal append failed"));
        }
    }

    Journal(const Journal&) = delete;
    Journal& operator=(const Journal&) = delete;

    ~Journal() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    const std::vector<CompletedRecord>& records() const { return records_; }

    void append(const CompletedRecord& record, bool synchronize) {
        validate_record_state(record, records_);
        const std::vector<std::uint8_t> body = journal_version_ == kJournalVersion1
            ? serialize_completed_v1(record) : serialize_completed(record);
        const auto digest = sha256_bytes(body.data(), body.size());
        std::vector<std::uint8_t> frame;
        append_le(frame, static_cast<std::uint32_t>(body.size()));
        frame.insert(frame.end(), body.begin(), body.end());
        frame.insert(frame.end(), digest.begin(), digest.end());
        append_le(frame, static_cast<std::uint32_t>(body.size()));
        write_all(fd_, frame.data(), frame.size());
        records_.push_back(record);
        if (synchronize) {
            sync();
        }
    }

    void sync() {
        if (::fsync(fd_) != 0) {
            throw std::runtime_error(errno_string("runtime journal fsync failed"));
        }
    }

  private:
    static constexpr std::size_t kHeaderV1Bytes = 8 + 4 + 32 * 3;
    static constexpr std::size_t kHeaderV2Bytes = 8 + 4 + 32;

    void validate_record_state(
        const CompletedRecord& record,
        const std::vector<CompletedRecord>& prior
    ) const {
        if (record.generated_tokens.size() > package_.max_new_tokens) {
            throw std::runtime_error("journal generated-token count exceeds package maximum");
        }
        for (const std::uint32_t token : record.generated_tokens) {
            if (token >= package_.embedding_rows) {
                throw std::runtime_error("journal generated token is outside the embedding table");
            }
        }
        if (!prior.empty()) {
            const auto& previous = prior.back().generated_tokens;
            if (record.generated_tokens.size() < previous.size() ||
                !std::equal(previous.begin(), previous.end(), record.generated_tokens.begin())) {
                throw std::runtime_error("journal generated-token history is not append-only");
            }
        }
        const bool terminal_tail = !record.generated_tokens.empty() &&
            std::find(
                package_.termination_tokens.begin(), package_.termination_tokens.end(),
                record.generated_tokens.back()
            ) != package_.termination_tokens.end();
        if (record.terminated != terminal_tail) {
            throw std::runtime_error("journal termination state differs from generated tokens");
        }
    }

    void create_header() {
        fs::create_directories(path_.parent_path());
        std::vector<std::uint8_t> header;
        journal_version_ = kJournalVersion2;
        const std::array<std::uint8_t, 8> magic = {'A','C','E','2','J','2',0,0};
        header.insert(header.end(), magic.begin(), magic.end());
        append_le(header, kJournalVersion2);
        header.insert(header.end(), package_.package_sha.begin(), package_.package_sha.end());
        const int fd = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
        if (fd < 0) {
            throw std::runtime_error(errno_string("create runtime journal failed"));
        }
        write_all(fd, header.data(), header.size());
        if (::fsync(fd) != 0) {
            const std::string message = errno_string("runtime journal header fsync failed");
            ::close(fd);
            throw std::runtime_error(message);
        }
        ::close(fd);
        fsync_directory(path_);
    }

    std::vector<CompletedRecord> load_existing() {
        std::vector<std::uint8_t> data = read_file(path_);
        if (data.size() < 12) {
            throw std::runtime_error("runtime journal header differs");
        }
        std::size_t offset = 8;
        journal_version_ = read_le<std::uint32_t>(data, offset);
        if (std::memcmp(data.data(), "ACE2J2\0\0", 8) == 0 &&
            journal_version_ == kJournalVersion2) {
            if (data.size() < kHeaderV2Bytes ||
                std::memcmp(data.data() + offset, package_.package_sha.data(),
                            package_.package_sha.size()) != 0) {
                throw std::runtime_error("runtime journal provenance differs");
            }
            offset += package_.package_sha.size();
        } else if (std::memcmp(data.data(), "ACE2J1\0\0", 8) == 0 &&
                   journal_version_ == kJournalVersion1 &&
                   package_.version == kPackageVersion1) {
            if (data.size() < kHeaderV1Bytes) {
                throw std::runtime_error("runtime journal header differs");
            }
            for (const auto* expected : {
                     &package_.schedule_sha, &package_.image_sha, &package_.model_sha
                 }) {
                if (std::memcmp(data.data() + offset, expected->data(), expected->size()) != 0) {
                    throw std::runtime_error("runtime journal provenance differs");
                }
                offset += expected->size();
            }
        } else {
            throw std::runtime_error("runtime journal version differs");
        }
        std::vector<CompletedRecord> records;
        std::size_t valid_bytes = offset;
        while (offset < data.size()) {
            if (data.size() - offset < 4) {
                break;
            }
            const std::size_t frame_start = offset;
            const std::uint32_t body_size = read_le<std::uint32_t>(data, offset);
            if (data.size() - offset < static_cast<std::size_t>(body_size) + 32 + 4) {
                break;
            }
            std::vector<std::uint8_t> body(
                data.begin() + static_cast<std::ptrdiff_t>(offset),
                data.begin() + static_cast<std::ptrdiff_t>(offset + body_size)
            );
            offset += body_size;
            const auto digest = sha256_bytes(body.data(), body.size());
            if (std::memcmp(data.data() + offset, digest.data(), digest.size()) != 0) {
                throw std::runtime_error("runtime journal frame digest differs");
            }
            offset += digest.size();
            const std::uint32_t trailing_size = read_le<std::uint32_t>(data, offset);
            if (trailing_size != body_size) {
                throw std::runtime_error("runtime journal frame trailer differs");
            }
            CompletedRecord record = journal_version_ == kJournalVersion1
                ? parse_completed_v1(body) : parse_completed(body);
            if (record.ordinal != records.size()) {
                throw std::runtime_error("runtime journal ordinal sequence differs");
            }
            validate_record_state(record, records);
            records.push_back(std::move(record));
            valid_bytes = offset;
            if (offset <= frame_start) {
                throw std::runtime_error("runtime journal parser made no progress");
            }
        }
        if (valid_bytes != data.size()) {
            if (::truncate(path_.c_str(), static_cast<off_t>(valid_bytes)) != 0) {
                throw std::runtime_error(errno_string("truncate incomplete journal tail failed"));
            }
            fsync_directory(path_);
        }
        return records;
    }

    fs::path path_;
    const Package& package_;
    int fd_ = -1;
    std::uint32_t journal_version_ = kJournalVersion2;
    std::vector<CompletedRecord> records_;
};

std::array<std::uint8_t, 16> get_wide128(const WData* words) {
    std::array<std::uint8_t, 16> bytes{};
    for (std::size_t word = 0; word < 4; ++word) {
        const std::uint32_t value = words[word];
        for (std::size_t lane = 0; lane < 4; ++lane) {
            bytes[word * 4 + lane] = static_cast<std::uint8_t>(
                (value >> (lane * 8)) & 0xffU
            );
        }
    }
    return bytes;
}

void set_wide128(WData* words, const std::array<std::uint8_t, 16>& bytes) {
    for (std::size_t word = 0; word < 4; ++word) {
        std::uint32_t value = 0;
        for (std::size_t lane = 0; lane < 4; ++lane) {
            value |= static_cast<std::uint32_t>(bytes[word * 4 + lane]) << (lane * 8);
        }
        words[word] = value;
    }
}

void sha_update_u64(SHA256_CTX& context, std::uint64_t value) {
    std::array<std::uint8_t, 8> bytes{};
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::uint8_t>((value >> (index * 8)) & 0xffU);
    }
    SHA256_Update(&context, bytes.data(), bytes.size());
}

void sha_update_u16(SHA256_CTX& context, std::uint16_t value) {
    const std::array<std::uint8_t, 2> bytes = {
        static_cast<std::uint8_t>(value & 0xffU),
        static_cast<std::uint8_t>((value >> 8) & 0xffU),
    };
    SHA256_Update(&context, bytes.data(), bytes.size());
}

struct CommandResult {
    std::uint64_t cycles = 0;
    std::uint32_t read_beats = 0;
    std::uint16_t done_tag = 0;
    bool done_error = false;
    bool saturation = false;
    std::array<std::uint8_t, 32> source_sha{};
    std::array<std::uint8_t, 32> destination_sha{};
    std::vector<WriteRecord> writes;
};

class Simulator {
  public:
    Simulator(
        Memory& memory,
        std::uint64_t timeout_cycles,
        std::uint64_t read_response_latency_cycles,
        std::optional<std::uint64_t> read_error_address,
        std::optional<std::uint64_t> read_tag_mismatch_address,
        std::optional<std::uint64_t> reset_before_write_address
    )
        : memory_(memory),
          timeout_cycles_(timeout_cycles),
          read_response_latency_cycles_(read_response_latency_cycles),
          read_error_address_(read_error_address),
          read_tag_mismatch_address_(read_tag_mismatch_address),
          reset_before_write_address_(reset_before_write_address) {
        if (read_response_latency_cycles_ < kMinimumReadResponseLatencyCycles) {
            throw std::runtime_error("read response latency is below the one-cycle minimum");
        }
        clear_inputs();
        top_.rst_ni = 0;
        for (int cycle = 0; cycle < 5; ++cycle) {
            step();
        }
        top_.rst_ni = 1;
        enable_shell();
    }

    std::uint64_t cycles() const { return cycles_; }

    CommandResult execute(const Command& command) {
        if (read_response_.has_value() || write_request_.has_value() ||
            write_response_.has_value()) {
            throw BoundaryError("memory_protocol", "command started with outstanding memory traffic");
        }
        SHA256_CTX source_context;
        SHA256_CTX destination_context;
        SHA256_Init(&source_context);
        SHA256_Init(&destination_context);
        active_source_ = &source_context;
        active_destination_ = &destination_context;
        active_writes_.clear();
        active_read_beats_ = 0;

        drive_command(command);
        top_.cmd_valid_i = 1;
        top_.cmd_done_ready_i = 1;
        const std::uint64_t start_cycle = cycles_;
        bool accepted = false;
        while (true) {
            const Event event = step();
            if (event.reset_triggered) {
                for (const WriteRecord& write : active_writes_) {
                    if (reset_before_write_address_.has_value() &&
                        write.address >= *reset_before_write_address_ &&
                        write.address < *reset_before_write_address_ + 64ULL) {
                        throw BoundaryError(
                            "reset_publication",
                            "reset test observed a completion-sidecar write"
                        );
                    }
                }
                active_source_ = nullptr;
                active_destination_ = nullptr;
                throw BoundaryError(
                    "reset_during_busy",
                    "external reset canceled the command before sidecar publication"
                );
            }
            if (event.cmd_fire) {
                if (accepted) {
                    throw BoundaryError("descriptor_interface", "command accepted more than once");
                }
                accepted = true;
                top_.cmd_valid_i = 0;
            }
            if (event.done_fire) {
                if (!accepted) {
                    throw BoundaryError("descriptor_interface", "completion preceded command acceptance");
                }
                CommandResult result;
                result.cycles = cycles_ - start_cycle;
                result.read_beats = active_read_beats_;
                result.done_tag = event.done_tag;
                result.done_error = event.done_error;
                result.saturation = event.saturation;
                result.writes = active_writes_;
                SHA256_Final(result.source_sha.data(), &source_context);
                SHA256_Final(result.destination_sha.data(), &destination_context);
                active_source_ = nullptr;
                active_destination_ = nullptr;
                top_.cmd_done_ready_i = 0;
                if (read_response_.has_value() || write_request_.has_value() ||
                    write_response_.has_value()) {
                    throw BoundaryError(
                        "memory_protocol",
                        "completion retired with outstanding memory traffic"
                    );
                }
                validate_result(command, result);
                return result;
            }
            if (cycles_ - start_cycle > timeout_cycles_) {
                top_.cmd_valid_i = 0;
                top_.cmd_done_ready_i = 0;
                active_source_ = nullptr;
                active_destination_ = nullptr;
                throw BoundaryError("timeout", "command exceeded the host timeout policy");
            }
        }
    }

  private:
    struct ReadResponse {
        std::array<std::uint8_t, 16> data{};
        std::uint8_t tag = 0;
        std::uint64_t ready_cycle = 0;
        bool error = false;
        bool tag_mismatch = false;
    };

    struct WriteRequest {
        std::uint64_t address = 0;
        std::uint8_t tag = 0;
    };

    struct WriteResponse {
        std::uint8_t tag = 0;
    };

    struct Event {
        bool cmd_fire = false;
        bool done_fire = false;
        bool reset_triggered = false;
        std::uint16_t done_tag = 0;
        bool done_error = false;
        bool saturation = false;
    };

    void clear_inputs() {
        top_.clk_i = 0;
        top_.rst_ni = 0;
        top_.csr_valid_i = 0;
        top_.csr_write_i = 0;
        top_.csr_addr_i = 0;
        top_.csr_wdata_i = 0;
        top_.csr_wstrb_i = 0;
        top_.csr_rready_i = 0;
        top_.cmd_valid_i = 0;
        top_.cmd_opcode_i = 0;
        top_.cmd_flags_i = 0;
        top_.cmd_layer_id_i = 0;
        top_.cmd_m_i = 0;
        top_.cmd_n_i = 0;
        top_.cmd_k_i = 0;
        top_.cmd_sequence_position_i = 0;
        top_.cmd_completion_tag_i = 0;
        top_.cmd_src0_addr_i = 0;
        top_.cmd_src1_addr_i = 0;
        top_.cmd_dst_addr_i = 0;
        top_.cmd_scale_addr_i = 0;
        top_.cmd_scratch_addr_i = 0;
        top_.mem_req_ready_i = 0;
        top_.mem_wready_i = 0;
        top_.mem_rvalid_i = 0;
        set_wide128(top_.mem_rdata_i, {});
        top_.mem_rtag_i = 0;
        top_.mem_rerror_i = 0;
        top_.mem_bvalid_i = 0;
        top_.mem_btag_i = 0;
        top_.mem_berror_i = 0;
        top_.cmd_done_ready_i = 0;
    }

    void enable_shell() {
        top_.csr_valid_i = 1;
        top_.csr_write_i = 1;
        top_.csr_addr_i = kCsrControl;
        top_.csr_wdata_i = 1;
        top_.csr_wstrb_i = 0xff;
        const std::uint64_t start = cycles_;
        while (true) {
            top_.clk_i = 0;
            drive_memory_inputs();
            top_.eval();
            const bool fire = top_.csr_valid_i && top_.csr_ready_o;
            const Event ignored = capture_and_rise();
            (void)ignored;
            if (fire) {
                break;
            }
            if (cycles_ - start > 1024) {
                throw BoundaryError("csr_interface", "shell enable CSR timed out");
            }
        }
        top_.csr_valid_i = 0;
        top_.csr_write_i = 0;
        top_.csr_wstrb_i = 0;
    }

    void drive_command(const Command& command) {
        top_.cmd_opcode_i = command.opcode;
        top_.cmd_flags_i = command.flags;
        top_.cmd_layer_id_i = command.layer_id;
        top_.cmd_m_i = command.m;
        top_.cmd_n_i = command.n;
        top_.cmd_k_i = command.k;
        top_.cmd_sequence_position_i = command.sequence_position;
        top_.cmd_completion_tag_i = command.completion_tag;
        top_.cmd_src0_addr_i = command.src0;
        top_.cmd_src1_addr_i = command.src1;
        top_.cmd_dst_addr_i = command.dst;
        top_.cmd_scale_addr_i = command.scale;
        top_.cmd_scratch_addr_i = command.scratch;
    }

    void drive_memory_inputs() {
        top_.mem_req_ready_i = (cycles_ % 17ULL) != 3ULL;
        top_.mem_wready_i = (cycles_ % 19ULL) != 5ULL;
        const bool read_response_ready =
            read_response_.has_value() && cycles_ >= read_response_->ready_cycle;
        top_.mem_rvalid_i = read_response_ready;
        if (read_response_ready) {
            set_wide128(top_.mem_rdata_i, read_response_->data);
            top_.mem_rtag_i = read_response_->tag ^
                (read_response_->tag_mismatch ? 1U : 0U);
        } else {
            set_wide128(top_.mem_rdata_i, {});
            top_.mem_rtag_i = 0;
        }
        top_.mem_rerror_i = read_response_ready && read_response_->error;
        top_.mem_bvalid_i = write_response_.has_value();
        top_.mem_btag_i = write_response_.has_value() ? write_response_->tag : 0;
        top_.mem_berror_i = 0;
    }

    Event capture_and_rise() {
        Event event;
        const bool req_fire = top_.mem_req_valid_o && top_.mem_req_ready_i;
        const bool write_data_fire = top_.mem_wvalid_o && top_.mem_wready_i;
        const bool read_response_fire = top_.mem_rvalid_i && top_.mem_rready_o;
        const bool write_response_fire = top_.mem_bvalid_i && top_.mem_bready_o;
        event.cmd_fire = top_.cmd_valid_i && top_.cmd_ready_o;
        event.done_fire = top_.cmd_done_valid_o && top_.cmd_done_ready_i;
        event.done_tag = top_.cmd_done_tag_o;
        event.done_error = top_.cmd_done_error_o;
        event.saturation = top_.cmd_done_saturation_seen_o;

        const bool req_write = top_.mem_req_write_o;
        const std::uint64_t req_address = top_.mem_req_addr_o;
        const std::uint16_t req_len = top_.mem_req_len_o;
        const std::uint8_t req_tag = top_.mem_req_tag_o;
        const std::uint8_t write_tag = top_.mem_wtag_o;
        const std::uint16_t write_strobe = top_.mem_wstrb_o;
        const auto write_data = get_wide128(top_.mem_wdata_o);

        top_.clk_i = 1;
        top_.eval();

        if (read_response_fire) {
            read_response_.reset();
        }
        if (write_response_fire) {
            write_response_.reset();
        }
        if (req_fire) {
            if (req_len != 1) {
                throw BoundaryError("memory_protocol", "shell emitted mem_req_len other than one");
            }
            if (req_write) {
                if (write_request_.has_value()) {
                    throw BoundaryError("memory_protocol", "overlapping write requests", req_address);
                }
                write_request_ = WriteRequest{req_address, req_tag};
            } else {
                if (read_response_.has_value()) {
                    throw BoundaryError("memory_protocol", "overlapping read requests", req_address);
                }
                const auto data = memory_.read_beat(req_address);
                read_response_ = ReadResponse{
                    data,
                    req_tag,
                    cycles_ + read_response_latency_cycles_,
                    read_error_address_.has_value() &&
                        req_address == *read_error_address_,
                    read_tag_mismatch_address_.has_value() &&
                        req_address == *read_tag_mismatch_address_,
                };
                if (active_source_ != nullptr) {
                    sha_update_u64(*active_source_, req_address);
                    SHA256_Update(active_source_, data.data(), data.size());
                    ++active_read_beats_;
                }
            }
        }
        if (write_data_fire) {
            if (!write_request_.has_value()) {
                throw BoundaryError("memory_protocol", "write data lacked a write request");
            }
            if (write_request_->tag != write_tag) {
                throw BoundaryError("memory_protocol", "write request/data tags differ");
            }
            memory_.write_beat(write_request_->address, write_data, write_strobe);
            WriteRecord write{write_request_->address, write_strobe, write_data};
            active_writes_.push_back(write);
            if (active_destination_ != nullptr) {
                sha_update_u64(*active_destination_, write.address);
                sha_update_u16(*active_destination_, write.strobe);
                SHA256_Update(active_destination_, write.data.data(), write.data.size());
            }
            if (write_response_.has_value()) {
                throw BoundaryError("memory_protocol", "overlapping write responses");
            }
            write_response_ = WriteResponse{write_tag};
            write_request_.reset();
        }
        if (event.done_fire && (read_response_.has_value() || write_request_.has_value() ||
                                write_response_.has_value())) {
            throw BoundaryError("memory_protocol", "completion overlapped an unretired response");
        }
        top_.clk_i = 0;
        top_.eval();
        ++cycles_;
        return event;
    }

    Event step() {
        top_.clk_i = 0;
        drive_memory_inputs();
        top_.eval();
        if (reset_before_write_address_.has_value() && !reset_triggered_ &&
            top_.mem_req_valid_o && top_.mem_req_write_o &&
            top_.mem_req_addr_o == *reset_before_write_address_) {
            if (read_response_.has_value() || write_request_.has_value() ||
                write_response_.has_value()) {
                throw BoundaryError(
                    "reset_protocol",
                    "reset trigger encountered outstanding memory traffic"
                );
            }
            top_.mem_req_ready_i = 0;
            top_.mem_wready_i = 0;
            top_.rst_ni = 0;
            top_.eval();
            Event event = capture_and_rise();
            top_.rst_ni = 1;
            top_.eval();
            reset_triggered_ = true;
            event.reset_triggered = true;
            return event;
        }
        return capture_and_rise();
    }

    static std::uint32_t expected_writes(const Command& command) {
        switch (command.opcode) {
            case 1: return command.n / 16;
            case 2: return command.n / 16 +
                ((command.flags & kDynamicScale32Flag) != 0 ? 4U : 0U);
            case 3: return command.n / 16;
            case 4: return 1;
            case 5: return 1;
            case 6: return command.k / 16;
            case 7: return command.n / 16;
            case 8: return command.n / 16;
            case 9: return command.flags == 6 ? command.k / 16 : 0;
            case 10: return command.n / 16 * 2 + 1;
            case 11: return 1152 / 16;
            default:
                throw BoundaryError("descriptor_interface", "runtime saw an unknown opcode");
        }
    }

    static void validate_result(const Command& command, const CommandResult& result) {
        if (result.done_tag != command.completion_tag) {
            throw BoundaryError("completion_tag", "completion tag differs from the descriptor");
        }
        const bool dynamic_rms_provisional_numeric_failure =
            result.done_error && !result.saturation && command.opcode == 2 &&
            (command.flags & kDynamicScale32Flag) != 0 &&
            result.writes.size() == command.n / 16;
        const std::uint32_t expected = result.done_error && !result.saturation &&
                !dynamic_rms_provisional_numeric_failure
            ? 0U
            : (dynamic_rms_provisional_numeric_failure
                ? command.n / 16
                : expected_writes(command));
        if (result.writes.size() != expected) {
            std::ostringstream detail;
            detail << "write count differs: expected=" << expected
                   << " actual=" << result.writes.size();
            throw BoundaryError("dropped_writes", detail.str());
        }
    }

    Memory& memory_;
    std::uint64_t timeout_cycles_;
    std::uint64_t read_response_latency_cycles_;
    std::optional<std::uint64_t> read_error_address_;
    std::optional<std::uint64_t> read_tag_mismatch_address_;
    std::optional<std::uint64_t> reset_before_write_address_;
    bool reset_triggered_ = false;
    Vace2_shell_runtime_harness top_;
    std::uint64_t cycles_ = 0;
    std::optional<ReadResponse> read_response_;
    std::optional<WriteRequest> write_request_;
    std::optional<WriteResponse> write_response_;
    SHA256_CTX* active_source_ = nullptr;
    SHA256_CTX* active_destination_ = nullptr;
    std::vector<WriteRecord> active_writes_;
    std::uint32_t active_read_beats_ = 0;
};

struct RuntimeState {
    std::int32_t argmax_token = -1;
    std::int32_t argmax_logit = -129;
    std::vector<std::uint32_t> generated_tokens;
    bool terminated = false;
    std::uint32_t embedding_token = 0;
    std::int32_t loaded_token_step = -1;
    std::string embedding_sha;
    std::uint64_t resume_warmup_cycles = 0;
};

using ExecutionCommands = std::vector<const Command*>;

std::string preload_embedding(
    Memory& memory,
    const EmbeddingSource& embeddings,
    std::uint64_t destination,
    std::uint32_t token,
    float scale
) {
    const auto row = embeddings.quantized_row(token, scale);
    memory.preload(destination, row.data(), row.size());
    return hex_digest(sha256_bytes(row.data(), row.size()));
}

void preload_rope_command(
    Memory& memory,
    const Package& package,
    const Command& command
) {
    if (std::string(command.operator_name()) != "rope_q" &&
        std::string(command.operator_name()) != "rope_k") {
        return;
    }
    const std::size_t expected_rope_bytes = package.version == kPackageVersion1
        ? 2 * kRopeRecordBytes
        : (package.prompt_tokens.size() + package.max_new_tokens - 1) * kRopeRecordBytes;
    if ((package.version != kPackageVersion1 && package.version != kPackageVersion2) ||
        package.rope_records.size() != expected_rope_bytes) {
        throw BoundaryError(
            "state_propagation",
            "runtime RoPE coefficient records are absent or unauthenticated"
        );
    }
    if (static_cast<std::size_t>(command.sequence_position) * kRopeRecordBytes >=
            package.rope_records.size() ||
        (command.n != 896 && command.n != 128)) {
        throw BoundaryError("descriptor_interface", "runtime RoPE preload geometry changed");
    }
    const std::uint8_t* record = package.rope_records.data() +
        static_cast<std::size_t>(command.sequence_position) * kRopeRecordBytes;
    const std::uint64_t position_stride = command.n == 128 ? 512ULL : 3584ULL;
    const std::uint64_t position_base = command.src1 +
        static_cast<std::uint64_t>(command.sequence_position) * position_stride;
    for (std::uint32_t beat = 0; beat < command.n / 16; ++beat) {
        std::array<std::uint8_t, 64> expanded{};
        const std::size_t pair_offset = static_cast<std::size_t>(beat & 1U) * 32;
        std::memcpy(expanded.data(), record + pair_offset, 32);
        std::memcpy(expanded.data() + 32, record + 64 + pair_offset, 32);
        memory.preload(position_base + static_cast<std::uint64_t>(beat) * 64,
                       expanded.data(), expanded.size());
    }
}

void update_argmax(
    const Package& package,
    const Command& command,
    const Memory& memory,
    RuntimeState& state
) {
    if (std::string(command.operator_name()) != "lm_head_tile") {
        return;
    }
    if (command.vocab_tile < 0) {
        throw BoundaryError("token_selection", "lm-head command lacks a vocab tile");
    }
    const auto logits = memory.read_range(command.dst, kLmTile);
    for (std::uint32_t lane = 0; lane < kLmTile; ++lane) {
        const std::int32_t token = command.vocab_tile * kLmTile + lane;
        const std::int32_t logit = static_cast<std::int8_t>(logits[lane]);
        if (state.argmax_token < 0 || logit > state.argmax_logit ||
            (logit == state.argmax_logit && token < state.argmax_token)) {
            state.argmax_token = token;
            state.argmax_logit = logit;
        }
    }
    if (static_cast<std::uint32_t>(command.vocab_tile) == kLastLmTile) {
        const std::uint32_t final_prompt_step = static_cast<std::uint32_t>(
            package.prompt_tokens.size() - 1
        );
        if (command.token_step < final_prompt_step) {
            return;
        }
        const std::size_t generated_index = command.token_step - final_prompt_step;
        if (generated_index != state.generated_tokens.size() ||
            state.generated_tokens.size() >= package.max_new_tokens ||
            state.argmax_token < 0 ||
            static_cast<std::uint32_t>(state.argmax_token) >= package.embedding_rows) {
            throw BoundaryError(
                "token_selection",
                "generated-token sequence is noncanonical or outside the package bound"
            );
        }
        const std::uint32_t token = static_cast<std::uint32_t>(state.argmax_token);
        state.generated_tokens.push_back(token);
        state.terminated = std::find(
            package.termination_tokens.begin(), package.termination_tokens.end(), token
        ) != package.termination_tokens.end();
    }
}

void preload_embedding_for_step(
    const Package& package,
    std::uint32_t token_step,
    Memory& memory,
    const EmbeddingSource& embeddings,
    std::uint64_t embedding_destination,
    float embedding_scale,
    RuntimeState& state
) {
    std::uint32_t token = 0;
    if (token_step < package.prompt_tokens.size()) {
        token = package.prompt_tokens[token_step];
    } else {
        const std::size_t generated_index = token_step - package.prompt_tokens.size();
        if (generated_index >= state.generated_tokens.size()) {
            throw BoundaryError(
                "token_selection",
                "generated-token feedback began before its argmax completed"
            );
        }
        token = state.generated_tokens[generated_index];
    }
    state.embedding_token = token;
    stage_dynamic_scale32_sidecar_for_step(memory, package, token_step);
    state.embedding_sha = preload_embedding(
        memory, embeddings, embedding_destination, token, embedding_scale
    );
    state.loaded_token_step = static_cast<std::int32_t>(token_step);
    state.argmax_token = -1;
    state.argmax_logit = -129;
}

void replay_records(
    const Package& package,
    const ExecutionCommands& execution_commands,
    const std::vector<CompletedRecord>& records,
    Memory& memory,
    const EmbeddingSource& embeddings,
    std::uint64_t embedding_destination,
    float embedding_scale,
    RuntimeState& state
) {
    for (const CompletedRecord& record : records) {
        const Command& command = *execution_commands.at(record.ordinal);
        if (state.loaded_token_step != command.token_step) {
            preload_embedding_for_step(
                package, command.token_step, memory, embeddings, embedding_destination,
                embedding_scale, state
            );
        }
        for (const auto& write : record.writes) {
            memory.write_beat(write.address, write.data, write.strobe);
        }
        state.argmax_token = record.argmax_token;
        state.argmax_logit = record.argmax_logit;
        state.generated_tokens = record.generated_tokens;
        state.terminated = record.terminated;
    }
}

void append_json_tokens(std::ostringstream& out, const std::vector<std::uint32_t>& tokens) {
    out << '[';
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (index != 0) out << ", ";
        out << tokens[index];
    }
    out << ']';
}

std::string command_json_line(
    const CompletedRecord& record,
    const Command& command
) {
    std::ostringstream out;
    out << "{\"ordinal\":" << record.ordinal
        << ",\"source_ordinal\":" << command.ordinal
        << ",\"token_step\":" << command.token_step
        << ",\"layer_id\":" << static_cast<unsigned>(command.layer_id)
        << ",\"operator\":\"" << command.operator_name() << "\""
        << ",\"cycles\":" << record.cycles
        << ",\"read_beats\":" << record.read_beats
        << ",\"write_beats\":" << record.write_beats
        << ",\"source_sha256\":\"" << hex_digest(record.source_sha) << "\""
        << ",\"destination_sha256\":\"" << hex_digest(record.destination_sha) << "\""
        << ",\"completion_tag\":" << record.done_tag
        << ",\"completion_error\":" << (record.done_error ? "true" : "false")
        << ",\"saturation_seen\":" << (record.saturation ? "true" : "false")
        << ",\"argmax_token\":" << record.argmax_token
        << ",\"argmax_logit\":" << record.argmax_logit
        << ",\"generated_token_after\":";
    if (!record.generated_tokens.empty()) {
        out << record.generated_tokens.back();
    } else {
        out << "null";
    }
    out << ",\"generated_token_ids_after\":";
    append_json_tokens(out, record.generated_tokens);
    out << ",\"terminated_after\":" << (record.terminated ? "true" : "false")
        << "}\n";
    return out.str();
}

void rebuild_jsonl(
    const fs::path& path,
    const ExecutionCommands& execution_commands,
    const std::vector<CompletedRecord>& records
) {
    std::string content;
    for (const auto& record : records) {
        content += command_json_line(record, *execution_commands.at(record.ordinal));
    }
    write_atomic(path, content);
}

class JsonlAppender {
  public:
    explicit JsonlAppender(const fs::path& path) {
        fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (fd_ < 0) {
            throw std::runtime_error(errno_string("open command JSONL failed"));
        }
    }

    JsonlAppender(const JsonlAppender&) = delete;
    JsonlAppender& operator=(const JsonlAppender&) = delete;

    ~JsonlAppender() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    void append(const std::string& line, bool synchronize) {
        write_all(fd_, line);
        if (synchronize) {
            sync();
        }
    }

    void sync() {
        if (::fsync(fd_) != 0) {
            throw std::runtime_error(errno_string("command JSONL fsync failed"));
        }
    }

  private:
    int fd_ = -1;
};

std::string progress_json(
    const Package& package,
    std::size_t execution_command_count,
    const std::string& execution_mode,
    const RuntimeState& state,
    std::uint32_t next_ordinal,
    std::uint64_t cycles,
    std::uint64_t prior_cumulative_cycles,
    const fs::path& journal,
    const std::string& status,
    unsigned resume_count
) {
    std::ostringstream out;
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"status\": \"" << json_escape(status) << "\",\n"
        << "  \"execution_mode\": \"" << json_escape(execution_mode) << "\",\n"
        << "  \"total_commands\": " << execution_command_count << ",\n"
        << "  \"package_commands\": " << package.commands.size() << ",\n"
        << "  \"next_ordinal\": " << next_ordinal << ",\n"
        << "  \"last_committed_ordinal\": ";
    if (next_ordinal == 0) {
        out << "null";
    } else {
        out << (next_ordinal - 1);
    }
    out << ",\n"
        << "  \"simulator_cycles\": " << cycles << ",\n"
        << "  \"simulator_cycles_scope\": \"current_process_segment\",\n"
        << "  \"segment_simulator_cycles\": " << cycles << ",\n"
        << "  \"prior_cumulative_simulator_cycles\": "
        << prior_cumulative_cycles << ",\n"
        << "  \"cumulative_simulator_cycles\": "
        << (prior_cumulative_cycles + cycles) << ",\n"
        << "  \"schedule_sha256\": \"" << hex_digest(package.schedule_sha) << "\",\n"
        << "  \"image_sha256\": \"" << hex_digest(package.image_sha) << "\",\n"
        << "  \"model_safetensors_sha256\": \"" << hex_digest(package.model_sha) << "\",\n"
        << "  \"embedding_token\": " << state.embedding_token << ",\n"
        << "  \"embedding_s8_sha256\": \"" << state.embedding_sha << "\",\n"
        << "  \"argmax\": {\"token_id\": " << state.argmax_token
        << ", \"logit_s8\": " << state.argmax_logit << "},\n"
        << "  \"generated_token_ids\": ";
    append_json_tokens(out, state.generated_tokens);
    out << ",\n"
        << "  \"terminated\": " << (state.terminated ? "true" : "false") << ",\n"
        << "  \"journal_bytes\": " << (fs::exists(journal) ? fs::file_size(journal) : 0)
        << ",\n"
        << "  \"resume_count\": " << resume_count << ",\n"
        << "  \"resume_warmup_cycles\": " << state.resume_warmup_cycles << ",\n"
        << "  \"resume_warmup_cycles_scope\": \"current_process_segment\",\n"
        << "  \"first_failure\": null\n"
        << "}\n";
    return out.str();
}

std::string summary_json(
    const Package& package,
    const ExecutionCommands& execution_commands,
    const std::string& execution_mode,
    const RuntimeState& state,
    const std::vector<CompletedRecord>& records,
    std::uint64_t cycles,
    std::uint64_t prior_cumulative_cycles,
    unsigned resume_count,
    const std::string& status
) {
    struct Aggregate {
        std::uint64_t commands = 0;
        std::uint64_t cycles = 0;
        std::uint64_t max_cycles = 0;
        std::uint64_t reads = 0;
        std::uint64_t writes = 0;
    };
    std::map<std::string, Aggregate> aggregates;
    for (const auto& record : records) {
        const Command& command = *execution_commands.at(record.ordinal);
        Aggregate& aggregate = aggregates[command.operator_name()];
        ++aggregate.commands;
        aggregate.cycles += record.cycles;
        aggregate.max_cycles = std::max(aggregate.max_cycles, record.cycles);
        aggregate.reads += record.read_beats;
        aggregate.writes += record.write_beats;
    }
    std::ostringstream out;
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"status\": \"" << json_escape(status) << "\",\n"
        << "  \"execution_mode\": \"" << json_escape(execution_mode) << "\",\n"
        << "  \"commands_completed\": " << records.size() << ",\n"
        << "  \"commands_total\": " << execution_commands.size() << ",\n"
        << "  \"package_commands\": " << package.commands.size() << ",\n"
        << "  \"simulator_cycles\": " << cycles << ",\n"
        << "  \"simulator_cycles_scope\": \"current_process_segment\",\n"
        << "  \"segment_simulator_cycles\": " << cycles << ",\n"
        << "  \"prior_cumulative_simulator_cycles\": "
        << prior_cumulative_cycles << ",\n"
        << "  \"cumulative_simulator_cycles\": "
        << (prior_cumulative_cycles + cycles) << ",\n"
        << "  \"generated_token_ids\": ";
    append_json_tokens(out, state.generated_tokens);
    out << ",\n"
        << "  \"terminated\": " << (state.terminated ? "true" : "false") << ",\n"
        << "  \"resume_count\": " << resume_count << ",\n"
        << "  \"resume_warmup_cycles\": " << state.resume_warmup_cycles << ",\n"
        << "  \"resume_warmup_cycles_scope\": \"current_process_segment\",\n"
        << "  \"operator_totals\": {\n";
    std::size_t index = 0;
    for (const auto& item : aggregates) {
        out << "    \"" << json_escape(item.first) << "\": {"
            << "\"commands\":" << item.second.commands
            << ",\"cycles\":" << item.second.cycles
            << ",\"max_cycles\":" << item.second.max_cycles
            << ",\"read_beats\":" << item.second.reads
            << ",\"write_beats\":" << item.second.writes << "}";
        out << (++index == aggregates.size() ? "\n" : ",\n");
    }
    out << "  },\n"
        << "  \"first_failure\": null\n"
        << "}\n";
    return out.str();
}

std::string failure_json(
    const Package& package,
    const RuntimeState& state,
    std::uint32_t execution_ordinal,
    const Command& command,
    const BoundaryError& error,
    std::uint64_t cycles,
    std::uint64_t prior_cumulative_cycles
) {
    std::ostringstream out;
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"status\": \"STOPPED_AT_FIRST_GENUINE_BOUNDARY\",\n"
        << "  \"ordinal\": " << execution_ordinal << ",\n"
        << "  \"source_ordinal\": " << command.ordinal << ",\n"
        << "  \"token_step\": " << command.token_step << ",\n"
        << "  \"layer_id\": " << static_cast<unsigned>(command.layer_id) << ",\n"
        << "  \"operator\": \"" << command.operator_name() << "\",\n"
        << "  \"category\": \"" << json_escape(error.category) << "\",\n"
        << "  \"detail\": \"" << json_escape(error.what()) << "\",\n"
        << "  \"address\": ";
    if (error.address.has_value()) {
        out << *error.address;
    } else {
        out << "null";
    }
    out << ",\n"
        << "  \"simulator_cycles\": " << cycles << ",\n"
        << "  \"simulator_cycles_scope\": \"current_process_segment\",\n"
        << "  \"segment_simulator_cycles\": " << cycles << ",\n"
        << "  \"prior_cumulative_simulator_cycles\": "
        << prior_cumulative_cycles << ",\n"
        << "  \"cumulative_simulator_cycles\": "
        << (prior_cumulative_cycles + cycles) << ",\n"
        << "  \"schedule_sha256\": \"" << hex_digest(package.schedule_sha) << "\",\n"
        << "  \"image_sha256\": \"" << hex_digest(package.image_sha) << "\",\n"
        << "  \"generated_token_ids\": ";
    append_json_tokens(out, state.generated_tokens);
    out << ",\n"
        << "  \"terminated\": " << (state.terminated ? "true" : "false") << "\n"
        << "}\n";
    return out.str();
}

std::vector<Command> resume_warmup_commands(
    const ExecutionCommands& execution_commands,
    std::uint32_t next_ordinal
) {
    if (next_ordinal >= execution_commands.size()) {
        return {};
    }
    const Command& target = *execution_commands[next_ordinal];
    if (std::string(target.operator_name()) != "attention_compose" || target.flags == 0) {
        return {};
    }
    std::uint32_t start = next_ordinal;
    while (start != 0) {
        --start;
        const Command& candidate = *execution_commands[start];
        if (std::string(candidate.operator_name()) == "attention_compose" &&
            candidate.token_step == target.token_step &&
            candidate.layer_id == target.layer_id &&
            candidate.query_head == target.query_head &&
            candidate.flags == 0) {
            std::vector<Command> warmup;
            for (std::uint32_t index = start; index < next_ordinal; ++index) {
                warmup.push_back(*execution_commands[index]);
            }
            return warmup;
        }
    }
    throw BoundaryError("state_propagation", "cannot find attention-compose replay anchor");
}

std::string package_inspection_json(const Package& package) {
    std::ostringstream out;
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"format\": \"ACE2RT" << package.version << "\",\n"
        << "  \"version\": " << package.version << ",\n"
        << "  \"commands\": " << package.commands.size() << ",\n"
        << "  \"prompt_token_ids\": ";
    append_json_tokens(out, package.prompt_tokens);
    out << ",\n"
        << "  \"max_new_tokens\": " << package.max_new_tokens << ",\n"
        << "  \"termination_token_ids\": ";
    append_json_tokens(out, package.termination_tokens);
    out << ",\n"
        << "  \"rope_position_count\": " << package.rope_records.size() / kRopeRecordBytes << ",\n"
        << "  \"rope_records_sha256\": \""
        << hex_digest(sha256_bytes(package.rope_records.data(), package.rope_records.size()))
        << "\",\n"
        << "  \"dynamic_scale32_enabled\": "
        << (package.dynamic_scale32_enabled ? "true" : "false") << ",\n"
        << "  \"dynamic_scale32_model_identity64\": \""
        << std::hex << std::setw(16) << std::setfill('0')
        << package.dynamic_scale32_model_identity << std::dec << "\",\n"
        << "  \"dynamic_scale32_sidecar_count\": "
        << package.dynamic_scale32_sidecars.size() << ",\n"
        << "  \"dynamic_scale32_sidecar_records_sha256\": \""
        << hex_digest(package.dynamic_scale32_sidecars_sha) << "\",\n"
        << "  \"dynamic_scale32_payload_addresses\": [";
    for (std::size_t index = 0; index < package.dynamic_scale32_sidecars.size(); ++index) {
        if (index != 0) out << ", ";
        out << package.dynamic_scale32_sidecars[index].payload_address;
    }
    out << "],\n"
        << "  \"embedding_rows\": " << package.embedding_rows << ",\n"
        << "  \"embedding_columns\": " << package.embedding_cols << ",\n"
        << "  \"embedding_offset\": " << package.embedding_offset << ",\n"
        << "  \"schedule_sha256\": \"" << hex_digest(package.schedule_sha) << "\",\n"
        << "  \"image_sha256\": \"" << hex_digest(package.image_sha) << "\",\n"
        << "  \"model_sha256\": \"" << hex_digest(package.model_sha) << "\",\n"
        << "  \"tokenizer_sha256\": \"" << hex_digest(package.tokenizer_sha) << "\",\n"
        << "  \"prompt_tokens_sha256\": \"" << hex_digest(package.prompt_tokens_sha) << "\",\n"
        << "  \"package_sha256\": \"" << hex_digest(package.package_sha) << "\"\n"
        << "}\n";
    return out.str();
}

std::string journal_inspection_json(const Journal& journal) {
    std::ostringstream out;
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"records\": " << journal.records().size() << ",\n"
        << "  \"generated_token_ids\": ";
    if (journal.records().empty()) {
        append_json_tokens(out, {});
        out << ",\n  \"terminated\": false\n";
    } else {
        const CompletedRecord& last = journal.records().back();
        append_json_tokens(out, last.generated_tokens);
        out << ",\n  \"terminated\": " << (last.terminated ? "true" : "false") << "\n";
    }
    out << "}\n";
    return out.str();
}

struct Arguments {
    fs::path package;
    fs::path image;
    fs::path model;
    fs::path output;
    std::uint32_t stop_after = 0;
    std::uint64_t timeout_cycles = kDefaultTimeoutCycles;
    std::uint64_t read_response_latency_cycles = kMinimumReadResponseLatencyCycles;
    std::optional<std::uint32_t> corrupt_staged_sidecar_byte;
    std::optional<std::uint64_t> read_error_address;
    std::optional<std::uint64_t> read_tag_mismatch_address;
    std::optional<std::uint64_t> reset_before_write_address;
    std::optional<std::uint32_t> mutate_frozen_address_ordinal;
    std::optional<std::uint32_t> mutate_frozen_low_flags_ordinal;
    std::optional<std::uint32_t> focus_ds32_token_step;
    std::optional<std::uint32_t> focus_layer_prefix_through_token_step;
    std::uint32_t focus_layer_prefix_count = 1;
    bool focus_layer0_legacy = false;
    bool focus_layer_prefix_explicit = false;
    bool focus_layer_prefix_count_explicit = false;
    bool resume = false;
    bool prefill_skip_intermediate_lm_head = false;
    std::uint32_t persistence_batch_commands = 1;
    bool inspect_package = false;
    fs::path inspect_journal;
    ace2_runtime_identity::IdentityProfileKind identity_profile =
        ace2_runtime_identity::IdentityProfileKind::Base;
    bool identity_profile_explicit = false;
};

Arguments parse_arguments(int argc, char** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        auto require_value = [&]() -> std::string {
            if (++index >= argc) {
                throw std::runtime_error("missing value after " + option);
            }
            return argv[index];
        };
        if (option == "--package") {
            arguments.package = require_value();
        } else if (option == "--image") {
            arguments.image = require_value();
        } else if (option == "--model") {
            arguments.model = require_value();
        } else if (option == "--output") {
            arguments.output = require_value();
        } else if (option == "--stop-after") {
            arguments.stop_after = static_cast<std::uint32_t>(std::stoul(require_value()));
        } else if (option == "--timeout-cycles") {
            arguments.timeout_cycles = std::stoull(require_value());
        } else if (option == "--read-response-latency-cycles") {
            arguments.read_response_latency_cycles = std::stoull(require_value());
        } else if (option == "--corrupt-staged-sidecar-byte") {
            arguments.corrupt_staged_sidecar_byte = static_cast<std::uint32_t>(
                std::stoul(require_value(), nullptr, 0)
            );
        } else if (option == "--read-error-address") {
            arguments.read_error_address = std::stoull(require_value(), nullptr, 0);
        } else if (option == "--read-tag-mismatch-address") {
            arguments.read_tag_mismatch_address =
                std::stoull(require_value(), nullptr, 0);
        } else if (option == "--reset-before-write-address") {
            arguments.reset_before_write_address =
                std::stoull(require_value(), nullptr, 0);
        } else if (option == "--rtl-mutate-frozen-address") {
            arguments.mutate_frozen_address_ordinal = static_cast<std::uint32_t>(
                std::stoul(require_value(), nullptr, 0)
            );
        } else if (option == "--rtl-mutate-frozen-low-flags") {
            arguments.mutate_frozen_low_flags_ordinal = static_cast<std::uint32_t>(
                std::stoul(require_value(), nullptr, 0)
            );
        } else if (option == "--focus-ds32-token-step") {
            arguments.focus_ds32_token_step = static_cast<std::uint32_t>(
                std::stoul(require_value(), nullptr, 0)
            );
        } else if (option == "--focus-layer0-through-token-step") {
            arguments.focus_layer_prefix_through_token_step = static_cast<std::uint32_t>(
                std::stoul(require_value(), nullptr, 0)
            );
            arguments.focus_layer0_legacy = true;
        } else if (option == "--focus-layer-prefix-through-token-step") {
            arguments.focus_layer_prefix_through_token_step = static_cast<std::uint32_t>(
                std::stoul(require_value(), nullptr, 0)
            );
            arguments.focus_layer_prefix_explicit = true;
        } else if (option == "--focus-layer-prefix-count") {
            arguments.focus_layer_prefix_count = static_cast<std::uint32_t>(
                std::stoul(require_value(), nullptr, 0)
            );
            arguments.focus_layer_prefix_count_explicit = true;
        } else if (option == "--resume") {
            arguments.resume = true;
        } else if (option == "--prefill-skip-intermediate-lm-head") {
            arguments.prefill_skip_intermediate_lm_head = true;
        } else if (option == "--persistence-batch-commands") {
            arguments.persistence_batch_commands = static_cast<std::uint32_t>(
                std::stoul(require_value())
            );
        } else if (option == "--inspect-package") {
            arguments.inspect_package = true;
        } else if (option == "--inspect-journal") {
            arguments.inspect_journal = require_value();
        } else if (option == "--identity-profile") {
            if (arguments.identity_profile_explicit) {
                throw std::runtime_error("--identity-profile may be specified only once");
            }
            const std::string profile_name = require_value();
            const auto parsed_profile =
                ace2_runtime_identity::parse_identity_profile(profile_name);
            if (!parsed_profile.has_value()) {
                throw std::runtime_error("unknown --identity-profile: " + profile_name);
            }
            arguments.identity_profile = *parsed_profile;
            arguments.identity_profile_explicit = true;
        } else {
            throw std::runtime_error("unknown runtime option: " + option);
        }
    }
    if (arguments.package.empty()) {
        throw std::runtime_error("--package is required");
    }
    if (!arguments.inspect_package && arguments.inspect_journal.empty() &&
        (arguments.image.empty() || arguments.model.empty() || arguments.output.empty())) {
        throw std::runtime_error("--package, --image, --model, and --output are required");
    }
    if (arguments.read_response_latency_cycles < kMinimumReadResponseLatencyCycles) {
        throw std::runtime_error("--read-response-latency-cycles must be at least one");
    }
    if (arguments.persistence_batch_commands == 0U) {
        throw std::runtime_error("--persistence-batch-commands must be at least one");
    }
    if (arguments.corrupt_staged_sidecar_byte.has_value() &&
        *arguments.corrupt_staged_sidecar_byte >= 64U) {
        throw std::runtime_error("--corrupt-staged-sidecar-byte must be in 0..63");
    }
    for (const auto ordinal : {arguments.mutate_frozen_address_ordinal,
                               arguments.mutate_frozen_low_flags_ordinal}) {
        if (ordinal.has_value() && *ordinal > 3U) {
            throw std::runtime_error("RTL DS32 tuple mutation ordinal must be zero through three");
        }
    }
    if (arguments.focus_ds32_token_step.has_value() && arguments.resume) {
        throw std::runtime_error("--focus-ds32-token-step is incompatible with --resume");
    }
    if (arguments.focus_ds32_token_step.has_value() &&
        arguments.focus_layer_prefix_through_token_step.has_value()) {
        throw std::runtime_error(
            "focused DS32 and focused layer-prefix modes are mutually exclusive"
        );
    }
    if (arguments.focus_layer_prefix_count < 1U ||
        arguments.focus_layer_prefix_count > kTransformerLayers) {
        throw std::runtime_error("--focus-layer-prefix-count must be between 1 and 24");
    }
    if (arguments.focus_layer0_legacy && arguments.focus_layer_prefix_explicit) {
        throw std::runtime_error(
            "legacy layer-0 and generic layer-prefix focus options are mutually exclusive"
        );
    }
    if (arguments.focus_layer0_legacy && arguments.focus_layer_prefix_count_explicit) {
        throw std::runtime_error(
            "--focus-layer-prefix-count is incompatible with the legacy layer-0 option"
        );
    }
    if (arguments.focus_layer_prefix_count_explicit &&
        !arguments.focus_layer_prefix_explicit) {
        throw std::runtime_error(
            "--focus-layer-prefix-count requires --focus-layer-prefix-through-token-step"
        );
    }
    if (arguments.prefill_skip_intermediate_lm_head &&
        (arguments.focus_ds32_token_step.has_value() ||
         arguments.focus_layer_prefix_through_token_step.has_value())) {
        throw std::runtime_error(
            "prefill LM-head skipping is incompatible with focused execution modes"
        );
    }
    return arguments;
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    try {
        std::fesetround(FE_TONEAREST);
        const Arguments arguments = parse_arguments(argc, argv);
        const auto* selected_identity_profile =
            ace2_runtime_identity::profile_for(arguments.identity_profile);
        if (selected_identity_profile == nullptr) {
            throw std::runtime_error("unknown runtime identity profile");
        }
        const Package package = load_package(arguments.package, *selected_identity_profile);
        if (arguments.inspect_package) {
            std::cout << package_inspection_json(package);
            return 0;
        }
        if (!arguments.inspect_journal.empty()) {
            Journal journal(arguments.inspect_journal, package, true);
            std::cout << journal_inspection_json(journal);
            return 0;
        }
        std::array<const Command*, 4> focused_ds32{{nullptr, nullptr, nullptr, nullptr}};
        if (arguments.focus_ds32_token_step.has_value()) {
            const std::uint32_t position = *arguments.focus_ds32_token_step;
            if (position >= package.prompt_tokens.size() ||
                arguments.stop_after == 0 || arguments.stop_after > 4) {
                throw std::runtime_error(
                    "--focus-ds32-token-step requires a prompt position and --stop-after 1..4"
                );
            }
            for (const Command& command : package.commands) {
                if (command.token_step == position && command.layer_id == 0 &&
                    command.operator_id <= 3) {
                    focused_ds32.at(command.operator_id) = &command;
                }
            }
            if (std::any_of(
                    focused_ds32.begin(), focused_ds32.end(),
                    [](const Command* command) { return command == nullptr; }
                )) {
                throw std::runtime_error("focused DS32 prompt tranche is absent");
            }
        }
        ExecutionCommands execution_commands;
        std::string execution_mode = "full_package_schedule";
        if (arguments.focus_ds32_token_step.has_value()) {
            execution_mode = "focused_ds32_prompt_position";
            execution_commands.assign(focused_ds32.begin(), focused_ds32.end());
        } else if (arguments.focus_layer_prefix_through_token_step.has_value()) {
            const std::uint32_t through =
                *arguments.focus_layer_prefix_through_token_step;
            const std::uint32_t layer_count = arguments.focus_layer_prefix_count;
            if (through >= package.prompt_tokens.size()) {
                throw std::runtime_error(
                    "--focus-layer-prefix-through-token-step requires a prompt position"
                );
            }
            execution_mode = layer_count == 1U
                ? "focused_complete_layer0_prompt_prefix"
                : "focused_complete_layer_prefix_prompt_prefix";
            for (const Command& command : package.commands) {
                if (command.token_step <= through && command.layer_id < layer_count) {
                    execution_commands.push_back(&command);
                }
            }
            if (execution_commands.empty() || execution_commands.front()->token_step != 0 ||
                execution_commands.back()->token_step != through) {
                throw std::runtime_error(
                    "focused complete layer-prefix prompt prefix is incomplete"
                );
            }
            for (std::uint32_t position = 0; position <= through; ++position) {
                for (std::uint32_t layer = 0; layer < layer_count; ++layer) {
                    const auto first = std::find_if(
                        execution_commands.begin(), execution_commands.end(),
                        [position, layer](const Command* command) {
                            return command->token_step == position &&
                                command->layer_id == layer;
                        }
                    );
                    const auto last = std::find_if(
                        execution_commands.rbegin(), execution_commands.rend(),
                        [position, layer](const Command* command) {
                            return command->token_step == position &&
                                command->layer_id == layer;
                        }
                    );
                    if (first == execution_commands.end() ||
                        last == execution_commands.rend() ||
                        std::string((*first)->operator_name()) != "input_rmsnorm" ||
                        std::string((*last)->operator_name()) != "mlp_residual_add") {
                        throw std::runtime_error(
                            "focused complete layer-prefix prompt prefix is incomplete"
                        );
                    }
                }
            }
        } else {
            execution_commands.reserve(package.commands.size());
            for (const Command& command : package.commands) {
                const bool skip_intermediate_prompt_lm_head =
                    arguments.prefill_skip_intermediate_lm_head &&
                    command.token_step + 1U < package.prompt_tokens.size() &&
                    std::string(command.operator_name()) == "lm_head_tile";
                if (!skip_intermediate_prompt_lm_head) {
                    execution_commands.push_back(&command);
                }
            }
            if (arguments.prefill_skip_intermediate_lm_head) {
                execution_mode = "prefill_skip_intermediate_lm_head";
                const std::size_t expected_skipped =
                    (package.prompt_tokens.size() - 1U) * kLmTileCount;
                if (package.commands.size() - execution_commands.size() != expected_skipped) {
                    throw std::runtime_error(
                        "prefill LM-head skip count differs from the frozen package schedule"
                    );
                }
                for (std::size_t position = 0;
                     position + 1U < package.prompt_tokens.size(); ++position) {
                    const auto last = std::find_if(
                        execution_commands.rbegin(), execution_commands.rend(),
                        [position](const Command* command) {
                            return command->token_step == position;
                        }
                    );
                    if (last == execution_commands.rend() ||
                        std::string((*last)->operator_name()) != "final_rmsnorm") {
                        throw std::runtime_error("prefill LM-head skip boundary differs");
                    }
                }
            }
        }
        const std::uint32_t stop_after = arguments.stop_after == 0
            ? static_cast<std::uint32_t>(execution_commands.size())
            : arguments.stop_after;
        if (stop_after > execution_commands.size()) {
            throw std::runtime_error("--stop-after exceeds execution command count");
        }
        fs::create_directories(arguments.output);
        const fs::path journal_path = arguments.output / "progress.journal";
        const fs::path jsonl_path = arguments.output / "commands.jsonl";
        const fs::path progress_path = arguments.output / "progress.json";
        const fs::path summary_path = arguments.output / "summary.json";
        const fs::path failure_path = arguments.output / "first_failure.json";
        if (!arguments.resume &&
            (fs::exists(jsonl_path) || fs::exists(progress_path) || fs::exists(summary_path) ||
             fs::exists(failure_path))) {
            throw std::runtime_error("fresh runtime output already contains progress artifacts");
        }
        std::uint64_t prior_cumulative_cycles = 0;
        unsigned prior_resume_count = 0;
        if (arguments.resume) {
            const auto summary_cumulative = read_json_unsigned_field(
                summary_path, "cumulative_simulator_cycles"
            );
            const auto progress_cumulative = read_json_unsigned_field(
                progress_path, "cumulative_simulator_cycles"
            );
            const auto summary_legacy = read_json_unsigned_field(
                summary_path, "simulator_cycles"
            );
            const auto progress_legacy = read_json_unsigned_field(
                progress_path, "simulator_cycles"
            );
            prior_cumulative_cycles = summary_cumulative.value_or(
                progress_cumulative.value_or(
                    summary_legacy.value_or(progress_legacy.value_or(0))
                )
            );
            prior_resume_count = static_cast<unsigned>(
                read_json_unsigned_field(progress_path, "resume_count").value_or(0)
            );
        }

        Memory memory(arguments.image, *selected_identity_profile);
        EmbeddingSource embeddings(arguments.model, package.embedding_offset);
        const auto scale_raw = memory.read_range(kRmsnormBase + kRmsnormScaleOffset, 8);
        double scale_f64 = 0.0;
        std::memcpy(&scale_f64, scale_raw.data(), sizeof(scale_f64));
        const float embedding_scale = static_cast<float>(scale_f64);
        if (!std::isfinite(embedding_scale) || embedding_scale <= 0.0f) {
            throw std::runtime_error("layer-0 embedding scale is invalid");
        }
        const std::uint64_t embedding_destination = package.commands.front().src0;

        RuntimeState state;
        const std::uint32_t initial_token_step =
            arguments.focus_ds32_token_step.value_or(0U);
        preload_embedding_for_step(
            package, initial_token_step, memory, embeddings, embedding_destination,
            embedding_scale, state
        );
        if (arguments.corrupt_staged_sidecar_byte.has_value()) {
            if (initial_token_step >= package.dynamic_scale32_sidecars.size()) {
                throw std::runtime_error(
                    "--corrupt-staged-sidecar-byte requires a staged DS32 prompt position"
                );
            }
            const DynamicScale32Sidecar& record =
                package.dynamic_scale32_sidecars.at(initial_token_step);
            const std::uint64_t address = record.payload_address - 64ULL +
                *arguments.corrupt_staged_sidecar_byte;
            auto value = memory.read_range(address, 1);
            value[0] ^= 1U;
            memory.preload(address, value.data(), value.size());
        }

        Journal journal(journal_path, package, arguments.resume);
        if (arguments.resume && !journal.records().empty() &&
            journal.records().back().done_error &&
            !journal.records().back().saturation) {
            throw std::runtime_error(
                "cannot resume past a journaled command-error boundary"
            );
        }
        if (journal.records().size() > execution_commands.size()) {
            throw std::runtime_error("journal progress exceeds selected execution schedule");
        }
        for (const CompletedRecord& record : journal.records()) {
            const Command& command = *execution_commands.at(record.ordinal);
            if (record.done_tag != command.completion_tag) {
                throw std::runtime_error(
                    "runtime journal execution policy differs from the selected schedule"
                );
            }
        }
        replay_records(
            package, execution_commands, journal.records(), memory, embeddings, embedding_destination,
            embedding_scale, state
        );
        if (arguments.resume) {
            rebuild_jsonl(jsonl_path, execution_commands, journal.records());
        } else {
            write_atomic(jsonl_path, "");
        }
        JsonlAppender jsonl(jsonl_path);
        if (journal.records().size() > stop_after) {
            throw std::runtime_error("journal progress exceeds requested stop ordinal");
        }

        Simulator simulator(
            memory,
            arguments.timeout_cycles,
            arguments.read_response_latency_cycles,
            arguments.read_error_address,
            arguments.read_tag_mismatch_address,
            arguments.reset_before_write_address
        );
        unsigned resume_count = arguments.resume ? prior_resume_count + 1U : 0U;
        const std::uint32_t next_ordinal = static_cast<std::uint32_t>(journal.records().size());
        if (arguments.resume) {
            const auto warmup = resume_warmup_commands(execution_commands, next_ordinal);
            const std::uint64_t start = simulator.cycles();
            for (const auto& command : warmup) {
                simulator.execute(command);
            }
            state.resume_warmup_cycles += simulator.cycles() - start;
        }
        write_atomic(
            progress_path,
            progress_json(
                package, execution_commands.size(), execution_mode, state, next_ordinal,
                simulator.cycles(), prior_cumulative_cycles, journal_path,
                "RUNNING", resume_count
            )
        );

        for (std::uint32_t ordinal = next_ordinal; ordinal < stop_after; ++ordinal) {
            const Command& command = *execution_commands.at(ordinal);
            Command dispatched_command = command;
            if (arguments.mutate_frozen_address_ordinal == ordinal) {
                if (ordinal == 0U) {
                    dispatched_command.dst += 0x1000ULL;
                } else {
                    dispatched_command.src0 += 0x1000ULL;
                }
            }
            if (arguments.mutate_frozen_low_flags_ordinal == ordinal) {
                dispatched_command.flags ^= 0x01U;
            }
            try {
                if (state.loaded_token_step != command.token_step) {
                    preload_embedding_for_step(
                        package, command.token_step, memory, embeddings,
                        embedding_destination, embedding_scale, state
                    );
                }
                preload_rope_command(memory, package, command);
                const CommandResult result = simulator.execute(dispatched_command);
                update_argmax(package, command, memory, state);
                CompletedRecord completed;
                completed.ordinal = ordinal;
                completed.cycles = result.cycles;
                completed.read_beats = result.read_beats;
                completed.write_beats = static_cast<std::uint32_t>(result.writes.size());
                completed.done_tag = result.done_tag;
                completed.done_error = result.done_error;
                completed.saturation = result.saturation;
                completed.argmax_token = state.argmax_token;
                completed.argmax_logit = state.argmax_logit;
                completed.generated_tokens = state.generated_tokens;
                completed.terminated = state.terminated;
                completed.source_sha = result.source_sha;
                completed.destination_sha = result.destination_sha;
                completed.writes = result.writes;
                const bool generation_done = package.version == kPackageVersion2 &&
                    (state.terminated ||
                     state.generated_tokens.size() >= package.max_new_tokens);
                const bool persistence_boundary =
                    (ordinal + 1U) % arguments.persistence_batch_commands == 0U ||
                    ordinal + 1U == stop_after || result.done_error || generation_done;
                journal.append(completed, persistence_boundary);
                jsonl.append(
                    command_json_line(completed, command),
                    persistence_boundary
                );
                if (result.done_error && !result.saturation) {
                    const BoundaryError error(
                        "command_error",
                        "shell completed the descriptor with cmd_done_error asserted"
                    );
                    write_atomic(
                        failure_path,
                        failure_json(
                            package, state, ordinal, command, error, simulator.cycles(),
                            prior_cumulative_cycles
                        )
                    );
                    write_atomic(
                        summary_path,
                        summary_json(
                            package, execution_commands, execution_mode, state,
                            journal.records(), simulator.cycles(), prior_cumulative_cycles, resume_count,
                            "STOPPED_AT_FIRST_GENUINE_BOUNDARY"
                        )
                    );
                    write_atomic(
                        progress_path,
                        progress_json(
                            package, execution_commands.size(), execution_mode, state,
                            ordinal + 1, simulator.cycles(), prior_cumulative_cycles, journal_path,
                            "STOPPED_AT_FIRST_GENUINE_BOUNDARY", resume_count
                        )
                    );
                    std::cerr << "ACE2_RUNTIME_BOUNDARY ordinal=" << ordinal
                              << " operator=" << command.operator_name()
                              << " category=" << error.category
                              << " detail=" << error.what() << '\n';
                    return 2;
                }
                const std::string status = ordinal + 1 == execution_commands.size() || generation_done
                    ? "COMPLETE" : "RUNNING";
                if (persistence_boundary) {
                    write_atomic(
                        progress_path,
                        progress_json(
                            package, execution_commands.size(), execution_mode, state,
                            ordinal + 1, simulator.cycles(), prior_cumulative_cycles, journal_path,
                            status, resume_count
                        )
                    );
                }
                const bool token_step_boundary = ordinal + 1 < execution_commands.size() &&
                    execution_commands[ordinal + 1]->token_step != command.token_step;
                if (ordinal < 2 || (ordinal + 1) % 100 == 0 ||
                    ordinal + 1 == stop_after || token_step_boundary || generation_done) {
                    std::cout << "ACE2_RUNTIME_PROGRESS next=" << (ordinal + 1)
                              << "/" << execution_commands.size()
                              << " operator=" << command.operator_name()
                              << " command_cycles=" << result.cycles
                              << " simulator_cycles=" << simulator.cycles() << '\n';
                    std::cout.flush();
                }
                if (generation_done) {
                    break;
                }
            } catch (const BoundaryError& error) {
                journal.sync();
                jsonl.sync();
                write_atomic(
                    failure_path,
                    failure_json(
                        package, state, ordinal, command, error, simulator.cycles(),
                        prior_cumulative_cycles
                    )
                );
                write_atomic(
                    summary_path,
                    summary_json(
                        package, execution_commands, execution_mode, state,
                        journal.records(), simulator.cycles(), prior_cumulative_cycles, resume_count,
                        "STOPPED_AT_FIRST_GENUINE_BOUNDARY"
                    )
                );
                write_atomic(
                    progress_path,
                    progress_json(
                        package, execution_commands.size(), execution_mode, state,
                        ordinal, simulator.cycles(), prior_cumulative_cycles, journal_path,
                        "STOPPED_AT_FIRST_GENUINE_BOUNDARY", resume_count
                    )
                );
                std::cerr << "ACE2_RUNTIME_BOUNDARY ordinal=" << ordinal
                          << " operator=" << command.operator_name()
                          << " category=" << error.category
                          << " detail=" << error.what() << '\n';
                return 2;
            }
        }

        const bool generation_done = package.version == kPackageVersion2 &&
            (state.terminated || state.generated_tokens.size() >= package.max_new_tokens);
        const bool full_complete = journal.records().size() == execution_commands.size() ||
            generation_done;
        const std::string final_status = full_complete ? "PASS" : "PREFIX_COMPLETE";
        write_atomic(
            summary_path,
            summary_json(
                package, execution_commands, execution_mode, state, journal.records(),
                simulator.cycles(), prior_cumulative_cycles, resume_count, final_status
            )
        );
        write_atomic(
            progress_path,
            progress_json(
                package, execution_commands.size(), execution_mode, state,
                static_cast<std::uint32_t>(journal.records().size()), simulator.cycles(),
                prior_cumulative_cycles, journal_path,
                final_status, resume_count
            )
        );
        std::cout << "ACE2_RUNTIME_" << final_status
                  << " commands=" << journal.records().size()
                  << " segment_simulator_cycles=" << simulator.cycles()
                  << " cumulative_simulator_cycles="
                  << (prior_cumulative_cycles + simulator.cycles());
        for (std::size_t index = 0; index < state.generated_tokens.size(); ++index) {
            std::cout << " token" << index << '=' << state.generated_tokens[index];
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ACE2_RUNTIME_SETUP_FAIL detail=" << error.what() << '\n';
        return 3;
    }
}
