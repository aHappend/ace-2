#ifndef ACE2_RUNTIME_IDENTITY_PROFILE_H_
#define ACE2_RUNTIME_IDENTITY_PROFILE_H_

#include <cstdint>
#include <optional>
#include <string_view>

namespace ace2_runtime_identity {

enum class IdentityProfileKind {
    Base,
    DiagnosticInstruct,
    SyntheticR6Descriptor,
    SyntheticR6RuntimeFixture,
    SyntheticR6RuntimeFixtureV2,
};

enum class ImageLayoutKind {
    Legacy,
    SyntheticR6FullScale32,
};

struct IdentityProfile {
    std::string_view name;
    std::string_view package_sha256;
    std::string_view image_sha256;
    std::string_view model_sha256;
    std::string_view tokenizer_sha256;
    std::uint64_t embedding_offset;
    bool require_exact_package_sha256;
    std::uint64_t image_bytes;
    ImageLayoutKind image_layout;
};

inline constexpr IdentityProfile kBaseIdentityProfile = {
    "base",
    "",
    "e24e0365e9fad5df2efe3e40df12e3f89f951f37c83449cb40e7d18fb614eafb",
    "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
    "f2e682984c6fbad1c922bdc0810c8c38b37f2f2be01aa588f3b4ed18e6e20bdb",
    32288,
    false,
    254421520,
    ImageLayoutKind::Legacy,
};

inline constexpr IdentityProfile kDiagnosticInstructIdentityProfile = {
    "diagnostic-instruct",
    "3e3eae19d69d0b24b7bdfd0b5eb0f35b1cf663d22554f988f6035b99f8191295",
    "ce1f94d930bf4d195b33c6aab4b18736419a34bb3671ea4d23ef2fe6be08ad42",
    "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "7c8c9883b532644facd612558ec66cdba192f3d0c5e6c7bc9fbca9fffaf6906a",
    32288,
    true,
    254421520,
    ImageLayoutKind::Legacy,
};

inline constexpr IdentityProfile kSyntheticR6DescriptorIdentityProfile = {
    "synthetic-r6-descriptor",
    "6c688a9131b927bb689b193e4ac2d250bd7039d7a83d923665e5c1adfe4f776d",
    "c784dbb734b52af806d9cf31db55f1fef708da1c5d9a34450c7c104c360e8835",
    "47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef",
    "7c8c9883b532644facd612558ec66cdba192f3d0c5e6c7bc9fbca9fffaf6906a",
    32288,
    true,
    254421520,
    ImageLayoutKind::Legacy,
};

inline constexpr IdentityProfile kSyntheticR6RuntimeFixtureIdentityProfile = {
    "synthetic-r6-runtime-fixture",
    "3ea5ab19742a0482ea16af800c742659319d477c52e26de72fbd4c9c9e6ae51b",
    "d2700046ede6a4400e653dc3e143ebb6dbc887d61276d220b53697f8fd337789",
    "f3e94a0d770ff6bbd0f44e3433130dbbcba4cb2fa6136918e899eac89aac6da9",
    "7c8c9883b532644facd612558ec66cdba192f3d0c5e6c7bc9fbca9fffaf6906a",
    32288,
    true,
    254421520,
    ImageLayoutKind::Legacy,
};

inline constexpr IdentityProfile kSyntheticR6RuntimeFixtureV2IdentityProfile = {
    "synthetic-r6-runtime-fixture-v2",
    "a982375bdceef8e5027e6b853853db8ebe2f8a9b612d723d4273e69b983928df",
    "ee404588126eea679d5ce3fdc73eb6062c699eec66e4c50b33549a204fc5b967",
    "f3e94a0d770ff6bbd0f44e3433130dbbcba4cb2fa6136918e899eac89aac6da9",
    "7c8c9883b532644facd612558ec66cdba192f3d0c5e6c7bc9fbca9fffaf6906a",
    32288,
    true,
    308869648,
    ImageLayoutKind::SyntheticR6FullScale32,
};

constexpr std::optional<IdentityProfileKind> parse_identity_profile(
    std::string_view name
) {
    if (name == kBaseIdentityProfile.name) {
        return IdentityProfileKind::Base;
    }
    if (name == kDiagnosticInstructIdentityProfile.name) {
        return IdentityProfileKind::DiagnosticInstruct;
    }
    if (name == kSyntheticR6DescriptorIdentityProfile.name) {
        return IdentityProfileKind::SyntheticR6Descriptor;
    }
    if (name == kSyntheticR6RuntimeFixtureIdentityProfile.name) {
        return IdentityProfileKind::SyntheticR6RuntimeFixture;
    }
    if (name == kSyntheticR6RuntimeFixtureV2IdentityProfile.name) {
        return IdentityProfileKind::SyntheticR6RuntimeFixtureV2;
    }
    return std::nullopt;
}

constexpr const IdentityProfile* profile_for(IdentityProfileKind kind) {
    switch (kind) {
        case IdentityProfileKind::Base:
            return &kBaseIdentityProfile;
        case IdentityProfileKind::DiagnosticInstruct:
            return &kDiagnosticInstructIdentityProfile;
        case IdentityProfileKind::SyntheticR6Descriptor:
            return &kSyntheticR6DescriptorIdentityProfile;
        case IdentityProfileKind::SyntheticR6RuntimeFixture:
            return &kSyntheticR6RuntimeFixtureIdentityProfile;
        case IdentityProfileKind::SyntheticR6RuntimeFixtureV2:
            return &kSyntheticR6RuntimeFixtureV2IdentityProfile;
    }
    return nullptr;
}

constexpr bool accepts_identity(
    const IdentityProfile& profile,
    std::string_view package_sha256,
    std::string_view image_sha256,
    std::string_view model_sha256,
    std::string_view tokenizer_sha256,
    std::uint64_t embedding_offset
) {
    return embedding_offset == profile.embedding_offset &&
        (!profile.require_exact_package_sha256 ||
         package_sha256 == profile.package_sha256) &&
        image_sha256 == profile.image_sha256 &&
        model_sha256 == profile.model_sha256 &&
        tokenizer_sha256 == profile.tokenizer_sha256;
}

}  // namespace ace2_runtime_identity

#endif  // ACE2_RUNTIME_IDENTITY_PROFILE_H_
