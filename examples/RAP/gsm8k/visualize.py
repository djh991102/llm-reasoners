import pickle
import sys
sys.path.append('..')
import os
from reasoners.visualization import visualize
from reasoners.visualization.tree_snapshot import NodeData
from reasoners.algorithm.mcts import MCTSNode
mcts_result = pickle.load(open('/mnt/nas/jaehyeok/llm-reasoners/logs/gsm8k_MCTS/11192024-173719/algo_output/1.pkl', 'rb'))
print(mcts_result.terminal_state)
def gsm_node_data_factory(x: MCTSNode):
    if not x.state:
        return {}
    return {"question": x.state[-1].sub_question, "answer": x.state[-1].sub_answer}
visualize(mcts_result, node_data_factory=gsm_node_data_factory)