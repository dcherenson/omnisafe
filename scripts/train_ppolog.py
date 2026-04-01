import argparse
import omnisafe

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--total-steps', type=int, default=2000000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-dir', type=str, default='../results')
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()

def main():
    args = parse_args()

    if args.env.startswith('SCG-'):
        import scg_wrapper

    custom_cfgs = {
        'seed': args.seed,
        'train_cfgs': {
            'total_steps': args.total_steps,
            'device': args.device,
        },
        'algo_cfgs': {
            'steps_per_epoch': 30000,
        },
        'logger_cfgs': {
            'log_dir': args.save_dir,
            'use_wandb': False,
            'use_tensorboard': True,
            'save_model_freq': 3,
        },
        'lagrange_cfgs': {
            'cost_limit': 25.0,
        },
        'model_cfgs': {
            'actor': {
                'hidden_sizes': [256, 256],
                'activation': 'tanh',
            },
            'critic': {
                'hidden_sizes': [256, 256],
                'activation': 'tanh',
            },
        },
    }

    print(f"Training PPO-Lag on {args.env} seed {args.seed}")
    agent = omnisafe.Agent('PPOLag', args.env, custom_cfgs=custom_cfgs)
    agent.learn()
    print(f"Done — results saved to {args.save_dir}")

if __name__ == '__main__':
    main()
