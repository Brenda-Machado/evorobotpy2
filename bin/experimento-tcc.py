import gym
import numpy as np
from policy import GymPolicyDiscr
import argparse
from tqdm import tqdm


def experimento_controle(env_name, ini_file, seed, n_episodios):
    """Combinação de todas as condições iniciais."""
    env = gym.make(env_name)
    policy = GymPolicyDiscr(env, ini_file, seed, test=0)
    resultados = []

    for i in range(n_episodios):
        recompensa, steps = policy.rollout(ntrials=1)
        print(f"Episódio {i+1} | Recompensa: {recompensa:.2f} | Passos: {steps}")
        resultados.append((i, recompensa, steps))

    env.close()
    return np.array(resultados)

def experimento_1(env_name, ini_file, seed, n_episodios):
    """Avaliação padrão por N episódios."""
    env = gym.make(env_name)
    policy = GymPolicyDiscr(env, ini_file, seed, test=0)
    resultados = []
    episodios = []

    for ep in episodios:
        for i in range(ep):
            recompensa, steps = policy.rollout(ntrials=1)
            print(f"Episódio {i+1} | Recompensa: {recompensa:.2f} | Passos: {steps}")
            resultados.append((i, recompensa, steps))

        env.close()
    # return np.array(resultados)


def experimento_2_duracao(env_name, ini_file, seed, duracao):
    """Varia a duração máxima do episódio."""
    env = gym.make(env_name)
    policy = GymPolicyDiscr(env, ini_file, seed, test=0)
    policy.maxsteps = duracao
    resultados = []

    for i in range(10):
        recompensa, steps = policy.rollout(ntrials=1)
        resultados.append((i, recompensa, steps))

    env.close()
    return np.array(resultados)


def experimento_3_ruido(env_name, ini_file, seed, noise_range=(-0.05, 0.05)):
    """Adiciona ruído à ação do agente."""
    env = gym.make(env_name)
    policy = GymPolicyDiscr(env, ini_file, seed, test=0)
    policy.action_noise = 1
    policy.action_noise_range = noise_range[1]  # a evonet usa apenas o valor máximo
    resultados = []

    for i in range(10):
        recompensa, steps = policy.rollout(ntrials=1)
        resultados.append((i, recompensa, steps))

    env.close()
    return np.array(resultados)


def experimento_4_iniciais(env_name, ini_file, seed, ranges):
    """Avalia com diferentes condições iniciais do estado."""
    env = gym.make(env_name)
    policy = GymPolicyDiscr(env, ini_file, seed, test=0)
    resultados = []

    for i in range(10):
        state = np.array([np.random.uniform(low, high) for low, high in ranges])
        env.reset()
        env.unwrapped.state = state
        recompensa, steps = policy.rollout(ntrials=1)
        resultados.append((i, *state, recompensa, steps))

    env.close()
    return np.array(resultados)


def experimento_5_fitness_peso(env_name, ini_file, seed, pesos=(0.3, 0.4, 0.3)):
    """Calcula fitness ponderado com base em recompensa mínima, média e máxima."""
    env = gym.make(env_name)
    policy = GymPolicyDiscr(env, ini_file, seed, test=0)
    recompensas = [policy.rollout(ntrials=1)[0] for _ in range(10)]
    env.close()

    r_min = np.min(recompensas)
    r_mean = np.mean(recompensas)
    r_max = np.max(recompensas)

    w_min, w_mean, w_max = pesos
    fitness = w_min * r_min + w_mean * r_mean + w_max * r_max
    return fitness


def run_all_experimentos(env_name, ini_file, seed):
    # print("Executando controle: combinação de condições iniciais")
    # np.save("resultados_exp1.npy", experimento_controle(env_name, ini_file, seed, n_episodios=10))

    print("Executando experimento 1: variação do número de episódios")
    np.save("resultados_exp1.npy", experimento_1(env_name, ini_file, seed, n_episodios=10))

    # print("Executando experimento 2: variação da duração de cada episódio")
    # np.save("resultados_exp2.npy", experimento_2_duracao(env_name, ini_file, seed, duracao=300))

    # print("Executando experimento 3: variação do ruído no motor do agente")
    # np.save("resultados_exp3.npy", experimento_3_ruido(env_name, ini_file, seed))

    # print("Executando experimento 4: variar intervalo das condições iniciais")
    # ranges = [(0, 0), (0, 0), (-0.2, 0.2), (-3.0, 3.0)]
    # np.save("resultados_exp4.npy", experimento_4_iniciais(env_name, ini_file, seed, ranges))

    # print("Executando experimento 5: avaliação do peso da fitness para cada episódio avaliativo")
    # fitness = experimento_5_fitness_peso(env_name, ini_file, seed)
    # print(f"Fitness ponderado: {fitness:.2f}")

    # print("Executando experimento 6: variação dos pesos da componente da fitness")
    # fitness = experimento_5_fitness_peso(env_name, ini_file, seed)
    # print(f"Fitness ponderado: {fitness:.2f}")


def run_experimento(env_name, ini_file, seed, n_episodios=10):
    env = gym.make(env_name)
    policy = GymPolicyDiscr(env, ini_file, seed, test=0)

    resultados = []

    for i in range(n_episodios):
        recompensa, steps = policy.rollout(ntrials=1)
        print(f"Episódio {i+1} | Recompensa: {recompensa:.2f} | Passos: {steps}")
        resultados.append((i, recompensa, steps))

    env.close()
    return np.array(resultados)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--ini", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodios", type=int, default=10)
    args = parser.parse_args()

    resultados = run_experimento(args.env, args.ini, args.seed, args.episodios)
    np.save("resultados_experimento.npy", resultados)
