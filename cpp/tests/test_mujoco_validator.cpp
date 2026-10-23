// The validator's pipeline and contact policy on a small world; the decision-parity corpus in Python is the oracle.
#include <mujoco/mujoco.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "harness.hpp"
#include "sscbirrt/mujoco/scene.hpp"
#include "sscbirrt/mujoco/snapshot.hpp"
#include "sscbirrt/mujoco/validator.hpp"

using namespace sscbirrt;
using sscbirrt::mujoco::Attachment;
using sscbirrt::mujoco::CollisionOptions;
using sscbirrt::mujoco::Scene;
using sscbirrt::mujoco::SceneValidator;
using sscbirrt::mujoco::Snapshot;

namespace {
const char* kXml = R"(
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1"/>
    <geom name="wall" type="box" pos="0.47 0 0.3" size="0.05 0.5 0.3"/>
    <body name="arm/base" pos="0 0 0.1">
      <joint name="j0" type="hinge" axis="0 0 1" limited="true" range="-3 3"/>
      <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.1"/>
      <body name="arm/link1" pos="0 0 0.1">
        <joint name="j1" type="hinge" axis="0 1 0" limited="true" range="-2 2"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0.4 0 0"/>
        <body name="arm/gripper/base" pos="0.4 0 0">
          <geom type="box" size="0.02 0.03 0.02"/>
          <body name="arm/gripper/finger" pos="0.04 0 0"><geom type="box" size="0.02 0.005 0.01"/></body>
        </body>
      </body>
    </body>
    <body name="can" pos="0 0.8 0.05"><freejoint/><geom type="cylinder" size="0.03 0.05"/></body>
    <body name="ball" mocap="true" pos="0 -0.8 0.5"><geom type="sphere" size="0.05"/></body>
  </worldbody>
</mujoco>)";

std::vector<std::byte> mjb() {
  static std::vector<std::byte> bytes = [] {
    const auto path = std::filesystem::temp_directory_path() / "sscbirrt_test_validator.xml";
    {
      std::ofstream f(path);
      f << kXml;
    }
    char err[1024] = {0};
    mjModel* m = mj_loadXML(path.string().c_str(), nullptr, err, sizeof err);
    if (!m) throw std::runtime_error(err);
    std::vector<std::byte> out(static_cast<std::size_t>(mj_sizeModel(m)));
    mj_saveModel(m, nullptr, out.data(), static_cast<int>(out.size()));
    mj_deleteModel(m);
    return out;
  }();
  return bytes;
}

std::shared_ptr<Scene> scene() { return std::make_shared<Scene>(mjb(), std::vector<std::string>{"j0", "j1"}); }

Snapshot rest(const Scene& s, std::array<double, 3> can_pos = {0.0, 0.8, 0.05}, std::array<double, 3> ball = {0.0, -0.8, 0.5}) {
  Snapshot snap;
  snap.qpos.assign(static_cast<std::size_t>(s.nq()), 0.0);
  snap.qpos[2] = can_pos[0];
  snap.qpos[3] = can_pos[1];
  snap.qpos[4] = can_pos[2];
  snap.qpos[5] = 1.0;  // can quaternion w
  snap.mocap_pos = {ball[0], ball[1], ball[2]};
  snap.mocap_quat = {1.0, 0.0, 0.0, 0.0};
  return snap;
}
}  // namespace

TEST(free_space_is_valid_and_the_wall_is_a_robot_environment_collision) {
  auto s = scene();
  SceneValidator v(s, rest(*s));
  CHECK(v.is_valid(Config{0.0, -1.2}));                     // arm pointing up and back
  CHECK(!v.is_valid(Config{0.0, 0.0}));                     // straight out along +x into the wall
  auto bad = v.invalid_contacts(Config{0.0, 0.0});
  CHECK(!bad.empty());
  for (const auto& c : bad) CHECK(c.kind == sscbirrt::mujoco::InvalidContact::Kind::RobotEnvironment);
}

TEST(floor_contact_and_mocap_obstacle_follow_the_snapshot) {
  auto s = scene();
  SceneValidator v(s, rest(*s));
  CHECK(!v.is_valid(Config{0.0, 1.4}));                     // arm swung down into the floor plane
  // Move the mocap ball into the arm's rest pose: the snapshot decides where obstacles are.
  SceneValidator blocked(s, rest(*s, {0.0, 0.8, 0.05}, {0.0, 0.0, 0.4}));
  CHECK(!blocked.is_valid(Config{0.0, -1.2}));
  CHECK(v.is_valid(Config{0.0, -1.2}));                     // the original validator is unaffected
}

TEST(attached_can_rides_with_the_gripper_and_allowed_contacts_are_not_collisions) {
  auto s = scene();
  const int can = s->body_id("can"), grip = s->body_id("arm/gripper/base");
  Attachment a;
  a.object_body = can;
  a.gripper_body = grip;
  a.T_gripper_object.at(0, 3) = 0.07;                        // the can sits between the fingers: it touches them, not the forearm
  a.allowed_bodies = s->subtree(grip);
  Snapshot snap = rest(*s);
  snap.attachments = {a};
  SceneValidator holding(s, snap);
  CHECK(holding.is_valid(Config{0.0, -1.2}));                // gripper-can contact is the grasp, allowed
  // With the finger not allowed, the same contact is a self-collision.
  Attachment strict = a;
  strict.allowed_bodies = {};
  Snapshot snap2 = rest(*s);
  snap2.attachments = {strict};
  SceneValidator not_allowed(s, snap2);
  auto bad = not_allowed.invalid_contacts(Config{0.0, -1.2});
  CHECK(!bad.empty());
  for (const auto& c : bad) CHECK(c.kind == sscbirrt::mujoco::InvalidContact::Kind::SelfCollision);
  // The held can against the wall is a robot-environment collision even when the arm itself is clear.
  Attachment far = a;
  far.T_gripper_object.at(0, 3) = 0.12;
  Snapshot snap3 = rest(*s);
  snap3.attachments = {far};
  SceneValidator carrying(s, snap3);
  const Config toward_wall{0.0, -0.6};                      // the arm clears the wall; the carried can does not
  SceneValidator empty_hand(s, rest(*s));
  CHECK(empty_hand.is_valid(toward_wall));
  CHECK(!carrying.is_valid(toward_wall));
}

TEST(repeated_queries_and_two_validators_agree_and_leak_nothing) {
  auto s = scene();
  for (int i = 0; i < 10; ++i) {
    SceneValidator a(s, rest(*s)), b(s, rest(*s));
    for (double q1 : {-1.2, -0.5, 0.0, 0.35, 1.4}) {
      const Config q{0.3 * i - 1.0, q1};
      CHECK(a.is_valid(q) == b.is_valid(q));
      CHECK(a.is_valid(q) == a.is_valid(q));
    }
  }
}

TEST(validators_at_two_clearances_share_one_scene_and_run_concurrently) {
  // #256: each validator owns its margins; the scene is shared read-only, so two clearances can run at once.
  auto s = scene();
  const std::string before = s->current_mjb_sha256();
  std::vector<Config> qs;
  for (int i = 0; i < 200; ++i) qs.push_back(Config{-1.5 + 0.015 * i, -0.6 + 0.004 * (i % 50)});
  auto answers = [&](double clearance) {
    SceneValidator v(s, rest(*s), CollisionOptions{clearance});
    std::vector<int> out;
    for (const Config& q : qs) out.push_back(v.is_valid(q) ? 1 : 0);
    return out;
  };
  const std::vector<int> zero = answers(0.0), wide = answers(0.05);
  CHECK(zero != wide);  // the clearance changes some answers, so independence is observable
  for (int round = 0; round < 4; ++round) {
    std::vector<int> got_zero, got_wide;
    std::thread a([&] { got_zero = answers(0.0); });
    std::thread b([&] { got_wide = answers(0.05); });
    a.join();
    b.join();
    CHECK(got_zero == zero);
    CHECK(got_wide == wide);
  }
  CHECK(s->current_mjb_sha256() == before && before == s->provenance().mjb_sha256);
  bool refused = false;
  try {
    SceneValidator bad(s, rest(*s), CollisionOptions{-0.001});
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  CHECK(refused);
}

HARNESS_MAIN()
