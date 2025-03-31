agent_config = {
    'lagrange' : {
        'cost_limit' : 10.0,
        'lambda_optimizer' : 'Adam',
        'lambda_lr' : 1e-3,
        'lambda_init' : 1.0,
        'lambda_upper_bound' : 10.0,
        'lagrange_start_step' : 1000
    },
    'q_net' : {
        'hidden_dims' : [256, 256],
        'act_fun' : 'relu',
        'out_act_fun' : 'identity',
        'opt_name' : 'Adam',
        'learning_rate' : 3e-4
    },
    'policy_net' : {
        'hidden_dims' : [256, 256],
        'act_fun' : 'relu',
        'out_act_fun' : 'identity',
        'out_std' : True,
        'conditioned_std' : True,
        'reparameter' : True,
        'log_std' : None,
        'log_std_min' : -20,
        'log_std_max' : 2,
        'stable_log_prob' : True,
        'opt_name' : 'Adam',
        'learning_rate' : 1e-5
    },
    'tau' : 0.005,
    'gamma' : 0.99,
    'update_interval' : 100,
    'alpha' : 0.5,
    'entropy' : {
        'auto_alpha' : True,
        'learning_rate' : 1e-3
    },
    'rand_times' : 5,
    'cql_temp' : 1.0,
    'cql_weight' : 5.0,
    'importance_sample' : False
}

trainer_config = {
    'cost_limit' : 10.0,
    'max_train_epoch' : 1000,
    'max_traj_len' : 1000,
    'eval_interval' : 10,
    'log_interval' : 5,
    'eval_traj_num' : 10,
    'batch_size' : 32,
    'model' : {
        'hidden_dims' : [200, 200, 200, 200],
        'batch_size' : 32,
        'learning_rate' : 1e-3,
        'reward_penalty_coef' : 0.0,
        'cost_penalty_coef' : 0.0,
        'hold_out_ratio' : 0.01,
        'max_model_update_epochs_to_improve': 5,
        'max_train_iteraions' : None,
        'rollout_length' : 3,
        'rollout_freq' : 2,
        'rollout_batch_size' : 1000,
        'real_ratio' : 0.8
    }
}