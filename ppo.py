__all__ = ['PerTokenPPOTrainer']

from trl.ppo import PPOTrainer

class PerTokenPPOTrainer(PPOTrainer):

    def __init__(self, model, ref_model, **ppo_params):
        super().__init__(model, ref_model, **ppo_params)

    def compute_rewards(self, scores, logprobs, ref_logprobs):
        """Compute per token rewards from scores and KL-penalty."""
        kl = logprobs - ref_logprobs # bs, respone_len
        non_score_reward = -self.kl_ctl.value * kl
        rewards = non_score_reward.clone().detach()
        rewards += scores
        return rewards, non_score_reward, self.kl_ctl.value
