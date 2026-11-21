"""调试：weld 激活的语义与稳定性。"""

import numpy as np

from arm_agent.env import ArmEnv


def show(env, tag):
    d = env.data
    m = env.model
    print(f"--- {tag} ---")
    print("  eq_active =", np.asarray(d.eq_active).tolist())
    for name, eq in env._eq_ids.items():
        print(f"  eq[{eq}] {name}: eq_data[0:7] = {np.round(m.eq_data[eq, 0:7], 5)}")
    print("  qacc finite:", np.all(np.isfinite(d.qacc)), " qpos finite:", np.all(np.isfinite(d.qpos)))


def main():
    env = ArmEnv()
    m, d = env.model, env.data

    print("n_eq =", m.neq, " eq_type =", m.eq_type.tolist())
    print("eq_obj1id =", m.eq_obj1id.tolist(), " eq_obj2id =", m.eq_obj2id.tolist())
    print("默认 eq_data[1] =", np.round(m.eq_data[1], 5).tolist())

    block = env.body_pos("red_block")
    grasp = block + np.array([0.0, 0.0, 0.02])
    ok, info = env.solve_to_pose(env.hand_target_for_grasp(grasp))
    print("到抓取点:", ok, "pos_err_mm=", round(info["ik_pos_err"] * 1000, 2))
    print("hand =", np.round(env.hand_pos(), 4), " grasp_point =", np.round(env.grasp_point(), 4))
    print("block =", np.round(env.body_pos("red_block"), 4))

    env.set_gripper(True)
    env.step(200)
    env.forward()
    print("闭爪后 block =", np.round(env.body_pos("red_block"), 4))

    env.attach("red_block")
    show(env, "attach 后立即")

    for i in range(1, 11):
        env.step(1)
        if not np.all(np.isfinite(d.qacc)):
            print(f"  第 {i} 步 QACC 出现非有限值")
            break
        if i % 2 == 0:
            print(f"  step{i}: block={np.round(env.body_pos('red_block'),4)} "
                  f"hand={np.round(env.hand_pos(),4)} maxqacc={np.abs(d.qacc).max():.1f}")

    show(env, "10 步之后")


if __name__ == "__main__":
    main()
