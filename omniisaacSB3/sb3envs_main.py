'''
Reference:
- https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_gym_new_rl_example.html
- D:\Common_Programs\omniverse\pkg\isaac_sim-2022.2.1\standalone_examples\api\omni.isaac.gym (train, play, task)

'''
import torch
from omni.isaac.gym.vec_env import VecEnvBase

if __name__ == '__main__':

    env = VecEnvBase(headless=False)
    from omniisaacSB3.basetasks.Cartpole import CartpoleTask
    task = CartpoleTask(name="Cartpole", n_envs=1)
    env.set_task(task, backend="torch")

    env._world.reset()
    obs = env.reset()
    while env._simulation_app.is_running():
        action = torch.randn(env.action_space.shape) #model.predict(obs)
        obs, rewards, dones, info = env.step(action)

    env.close()