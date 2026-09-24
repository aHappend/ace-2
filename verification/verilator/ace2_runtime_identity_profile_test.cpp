#include "ace2_runtime_identity_profile.h"

using ace2_runtime_identity::IdentityProfileKind;
using ace2_runtime_identity::accepts_identity;
using ace2_runtime_identity::kBaseIdentityProfile;
using ace2_runtime_identity::kDiagnosticInstructIdentityProfile;
using ace2_runtime_identity::parse_identity_profile;
using ace2_runtime_identity::profile_for;

constexpr auto kParsedBase = parse_identity_profile("base");
constexpr auto kParsedInstruct = parse_identity_profile("diagnostic-instruct");

static_assert(kParsedBase.has_value());
static_assert(*kParsedBase == IdentityProfileKind::Base);
static_assert(kParsedInstruct.has_value());
static_assert(*kParsedInstruct == IdentityProfileKind::DiagnosticInstruct);
static_assert(!parse_identity_profile("instruct").has_value());
static_assert(!parse_identity_profile("unknown").has_value());

static_assert(profile_for(IdentityProfileKind::Base) == &kBaseIdentityProfile);
static_assert(
    profile_for(IdentityProfileKind::DiagnosticInstruct) ==
    &kDiagnosticInstructIdentityProfile
);
static_assert(profile_for(static_cast<IdentityProfileKind>(255)) == nullptr);

static_assert(accepts_identity(
    kBaseIdentityProfile,
    "0000000000000000000000000000000000000000000000000000000000000000",
    "e24e0365e9fad5df2efe3e40df12e3f89f951f37c83449cb40e7d18fb614eafb",
    "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
    "f2e682984c6fbad1c922bdc0810c8c38b37f2f2be01aa588f3b4ed18e6e20bdb",
    32288
));

static_assert(accepts_identity(
    kDiagnosticInstructIdentityProfile,
    "3e3eae19d69d0b24b7bdfd0b5eb0f35b1cf663d22554f988f6035b99f8191295",
    "ce1f94d930bf4d195b33c6aab4b18736419a34bb3671ea4d23ef2fe6be08ad42",
    "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "7c8c9883b532644facd612558ec66cdba192f3d0c5e6c7bc9fbca9fffaf6906a",
    32288
));

static_assert(!accepts_identity(
    kDiagnosticInstructIdentityProfile,
    "0000000000000000000000000000000000000000000000000000000000000000",
    "ce1f94d930bf4d195b33c6aab4b18736419a34bb3671ea4d23ef2fe6be08ad42",
    "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "7c8c9883b532644facd612558ec66cdba192f3d0c5e6c7bc9fbca9fffaf6906a",
    32288
));
static_assert(!accepts_identity(
    kDiagnosticInstructIdentityProfile,
    "3e3eae19d69d0b24b7bdfd0b5eb0f35b1cf663d22554f988f6035b99f8191295",
    "e24e0365e9fad5df2efe3e40df12e3f89f951f37c83449cb40e7d18fb614eafb",
    "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "7c8c9883b532644facd612558ec66cdba192f3d0c5e6c7bc9fbca9fffaf6906a",
    32288
));
static_assert(!accepts_identity(
    kDiagnosticInstructIdentityProfile,
    "3e3eae19d69d0b24b7bdfd0b5eb0f35b1cf663d22554f988f6035b99f8191295",
    "ce1f94d930bf4d195b33c6aab4b18736419a34bb3671ea4d23ef2fe6be08ad42",
    "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
    "7c8c9883b532644facd612558ec66cdba192f3d0c5e6c7bc9fbca9fffaf6906a",
    32288
));
static_assert(!accepts_identity(
    kDiagnosticInstructIdentityProfile,
    "3e3eae19d69d0b24b7bdfd0b5eb0f35b1cf663d22554f988f6035b99f8191295",
    "ce1f94d930bf4d195b33c6aab4b18736419a34bb3671ea4d23ef2fe6be08ad42",
    "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "f2e682984c6fbad1c922bdc0810c8c38b37f2f2be01aa588f3b4ed18e6e20bdb",
    32288
));
static_assert(!accepts_identity(
    kDiagnosticInstructIdentityProfile,
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    32288
));

int main() {
    return 0;
}
