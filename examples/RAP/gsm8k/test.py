import pickle
# import sys
# sys.path.append('..')
import os
from reasoners.visualization import visualize
from reasoners.visualization.tree_snapshot import NodeData, TreeSnapshot, EdgeData, NodeId, EdgeId
from reasoners.algorithm.mcts import MCTSNode


if __name__ == '__main__':
    result_path = '/mnt/nas/jaehyeok/llm-reasoners/logs/gsm8k_MCTS/11192024-181238/algo_output/9.pkl'
    mcts_results = pickle.load(open(result_path, 'rb'))
    tree_states = mcts_results.tree_state_after_each_iter
    def node_data_factory(x: MCTSNode):
        if not x.state:
            return {}
        return {"question": x.state[-1].sub_question, "answer": x.state[-1].sub_answer}
    def get_reward_details(n: MCTSNode):
        if hasattr(n, "reward_details"):
            return n.reward_details
        return n.fast_reward_details if hasattr(n, "fast_reward_details") else None
    def edge_data_factory(n: MCTSNode):
        return EdgeData({"Q": n.Q, "reward": n.reward, **get_reward_details(n)})
    def all_nodes(node: MCTSNode):
        node_id = NodeId(node.id)

        nodes[node_id] = TreeSnapshot.Node(node_id, node_data_factory(node))
        if node.children is None:
            return
        for child in node.children:
            edge_id = EdgeId(len(edges))
            edges.append(TreeSnapshot.Edge(edge_id, node.id, child.id, edge_data_factory(child)))
            all_nodes(child)
    snapshots = []
    tree_states = tree_states[-1:]
    for step in range(len(tree_states)):
        edges = []
        nodes = {}

        root = tree_states[step]
        all_nodes(root)
        tree = TreeSnapshot(list(nodes.values()), edges)
        # select edges following the MCTS trace
        if mcts_results.trace_in_each_iter:
            trace = mcts_results.trace_in_each_iter[step]
            for step_idx in range(len(trace) - 1):
                in_node_id = trace[step_idx].id
                out_node_id = trace[step_idx + 1].id
                for edges in tree.out_edges(in_node_id):
                    if edges.target == out_node_id:
                        nodes[in_node_id].selected_edge = edges.id
                        break

        # for all other nodes, select edges with highest Q
        for node in tree.nodes.values():
            if node.selected_edge is None and tree.children(node.id):
                node.selected_edge = max(
                    tree.out_edges(node.id),
                    key=lambda edge: edge.data.get("Q", -float("inf"))
                ).id
        
        snapshots.append(tree)

    tree = snapshots[-1]
    root = tree.node(0)
    queue = [root]

    # def get_node_traj(curr):

    while len(queue) > 0:
        curr = queue.pop(0)
        if 'question' in curr.data:
            print(f"NODE ID: {curr.id}")
            print(f"QUESTION: {curr.data['question']}", '\n', '='*20)
            print(f"ANSWER: {curr.data['answer']}", '\n', '='*20)
            input()
        if curr.id in tree._children and len(tree._children[curr.id]) > 0:
            for child_id in list(tree._children[curr.id]):
                queue.append(tree.node(child_id))
        