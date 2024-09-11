from stable_baselines3.common.callbacks import EvalCallback


def create_eval_callback(env, best_model_save_path, log_path, eval_freq=2000):
    return EvalCallback(
        env=env,
        best_model_save_path=best_model_save_path,
        log_path=log_path,
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
