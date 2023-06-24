import torch 



if __name__ == '__main__':

    param_path = r"F:\zzm_codes\code2023\OmniIsaacGymEnvs\omniisaacgymenvs\runs\HumanoidPPO-7002\nn\Humanoid.pth"
    params = torch.load(param_path)
    # print(params.keys())
    params['model']