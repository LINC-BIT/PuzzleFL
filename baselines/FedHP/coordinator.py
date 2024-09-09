import copy
import numpy as np


class Coordinator:
    def __init__(self, client_ids) -> None:
        self.clients = {}
        for cid in client_ids:
            self.client_status[cid] = {}
        
        self.Ah_list = []
        self.miu_l_list = []
        self.tau_l_list = []
    
    def recieve_info_from_worker(self, cid, training_status):
        """接收来自worker的信息

        Args:
            cid (_type_): _description_
            training_status (_type_): 包含了计算时间mu_h_i, 通信时间beta_h_i_j_Nei, 共识距离D_h_i_j_Nei, L_i, sigma_i
        """
        self.client_status[cid] = training_status
    
    def send_info_to_worker(self, cid, round_id):
        Ah = self.Ah_list[round_id]
        tau_i = self.tau_l_list[round_id] / self.client_status[cid]['mu_h_i']   # TODO: 检测是否向下取整
        return Ah, tau_i
    
    def min_Ti(self, Ah):
        """ TODO: 根据拓扑矩阵Ah，计算得到

        Args:
            Ah (_type_): 拓扑矩阵，邻接矩阵，表示worker之间的连接关系

        Returns:
            l: 能够使总训练时间最小的worker的id
        """
        tau = 0
        self.tau_l_list.append(tau)
        
        return l 
    
    def adaptvie_control(self, ):
        N = workers = len(self.client_status.keys())
        A_h = np.ones(N, N)
        step = N
        flag = True 
        
        # 获得最小的完成时间Tl在base_topology下，以及在第h个round下的更新频率tau_l，
        l, Ti, tau = self.min_Ti(A_h)
        
        while True:
            if flag:
                step = np.sqrt(np.sum(A_h))
            else:
                step = np.ceil(step / 2)
            
            # 选择最慢的link在阈值下，加入到边集合E中
            A_prime = copy.deepcopy(A_h)
            E = self.get_slowest_links(A_prime, step)
            A_prime = self.remove_slowest_links(A_prime, step)
            
            # 重新计算最小的完成时间Tl在base_topology下，以及在第h个round下的更新频率tau_l，
            l_prime, Ti_prime, tau_prime = self.min_Ti(A_h)
            
            if Ti_prime < Ti:
                l, Ti, tau, Ah, flag = l_prime, Ti_prime, tau_prime, A_pri True
            else:
                flag = False
                
            if not flag and step == 1:
                break
        
        for i in self.client_status.keys():
            self.tau_i_h_list[i] = self.tau_l_list[i] / self.client_status[i]['mu_h_i']
        
        return self.tau_i_h_list, Ah
    
    
# adaptive control
def adaptive_control_algorithm(workers, mu_h_i_list, beta_h_i_j, D_h_i_j, L, sigma):
    # 初始化参数
    A_h = np.ones(len(workers), len(workers))
    s = len(workers)
    Flag = True
    
    # 计算初始训练时间
    T_i, tau_h_i = calculate_initial_training_time(workers, A_h)
    
    # 贪婪搜索最优拓扑和频率
    while True:
        if Flag:
            s = adjust_search_step(A_h)
        else:
            s = s // 2
        
        # 移除慢链接
        E = remove_slowest_links(A_h, s)
        A_prime = update_topology(A_h, E)
        
        # 重新计算训练时间
        T_i_prime, tau_h_i_prime = calculate_new_training_time(workers, A_prime)
        
        # 更新拓扑和频率
        if T_i_prime < T_i:
            A_h, tau_h_i = A_prime, tau_h_i_prime
            Flag = True
        else:
            Flag = False
        
        # 检查终止条件
        if not Flag and s == 1:
            break
    
    # 根据计算和通信时间更新频率
    tau_h_i = update_frequency(tau_h_i, T_i, workers)
    
    # 输出最终的频率和拓扑
    return tau_h_i, A_h