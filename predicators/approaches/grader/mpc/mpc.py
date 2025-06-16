import numpy as np
from predicators.approaches.grader.mpc.optimizers import RandomOptimizer
import copy
from predicators.structs import State
from predicators.envs.blocks import BlocksSimpleEnv

class MPC_Blocks(object):
    def __init__(self, mpc_args):
        self.type = mpc_args['type']
        self.horizon = mpc_args['horizon']
        self.gamma = mpc_args['gamma']
        self.popsize = mpc_args['popsize']

        # parameters from the environment
        self.action_dim = mpc_args['env_params']['action_dim']

    def act(self, task, model, state, possible_ent_names,
                      node_feature_to_index, action_list,
                      rng):
        # process the state to get pure state and goal
        pure_state = state[:len(state)] # remove the goal info at very beginning
        self.model = model
        self.state = pure_state

        # shoot random actions
        actions = rng.choice(action_list, size=(self.popsize, self.horizon))
        preprocessed_state = np.repeat(self.state[None], self.popsize, axis=0)
        action_idx = None
        for t_i in range(self.horizon):
            action = actions[:, t_i]
            # from int to one-hot
            action = np.eye(self.action_dim)[action]
            state_next = self.model.predict(preprocessed_state, action) + preprocessed_state
            goal_achieved = self.evaluate_goal_achieved(state_next, task, possible_ent_names,
                      node_feature_to_index)
            if goal_achieved.any():
                action_idx = np.where(goal_achieved)[0]
        # task the first step as our action
        if action_idx is None:
            action_idx = [0]
        action = actions[action_idx[0]]
        return action[0]

    def evaluate_goal_achieved(self, states, task, possible_ent_names,
                        node_feature_to_index):
        goal_achieved = np.zeros(self.popsize,)
        # for each vector state, convert to env task
        data = {}
        objs = list(task.init)
        for n in range(len(states)):
            state = states[n]
            for i, obj_name in enumerate(possible_ent_names):
                node_feat = state[i*len(node_feature_to_index):(i+1)*len(node_feature_to_index)]
                data_x = node_feat[node_feature_to_index["feat_pose_x"]]
                data_y = node_feat[node_feature_to_index["feat_pose_y"]]
                data_z = node_feat[node_feature_to_index["feat_pose_z"]]
                data_held = node_feat[node_feature_to_index["feat_held"]]
                data_finger = node_feat[node_feature_to_index["feat_fingers"]]
                data_r = node_feat[node_feature_to_index["feat_color_r"]]
                data_g = node_feat[node_feature_to_index["feat_color_g"]]
                data_b = node_feat[node_feature_to_index["feat_color_b"]]
                for obj in objs:
                    assert obj.name in possible_ent_names
                    if obj.name == obj_name:
                        if 'rob' in obj_name:
                            array_data = np.array([data_x, data_y, data_z, data_finger])
                            array_data = array_data.clip(0, 1)
                            data[obj] = array_data
                        else:
                            assert 'block' in obj_name
                            array_data = np.array([data_x, data_y, data_z, data_held, data_r, data_g, data_b])
                            array_data = array_data.clip(0, 1)
                            data[obj] = array_data
            goal_achieved[n] = task.goal_holds(State(data))
        return goal_achieved
