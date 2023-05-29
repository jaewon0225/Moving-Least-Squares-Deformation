import numpy as np
import matplotlib.pyplot as plt
def mls_affine_transform(p_input,q_input,v_input,a):
    ##Calculate weights w
    #convert p to nparray
    p,q,v = np.array(p_input), np.array(q_input), np.array(v_input)
    extruded_p = np.repeat(p[:, :, np.newaxis], len(v_input), axis=2) #[p,2,v]
    extruded_q = np.repeat(q[:, :, np.newaxis], len(v_input), axis=2) #[p,2,v]

    v = v.reshape(len(v_input),2)
    v = v.transpose() #[2,v]
    #calculate w
    w = 1/((np.sum((extruded_p-v)**2,axis=1))**a) #[p,v]

    ##Calculate p_star and q_star
    w = w.reshape(len(p_input),1,len(v_input))
    w_sum = np.sum(w, axis=0)
    w_sum.reshape(1, len(v_input))
    p_star = np.sum(extruded_p*w,axis=0)/w_sum
    q_star = np.sum(extruded_q*w,axis=0)/w_sum
    p_hat = extruded_p-p_star
    q_hat = extruded_q-q_star
    A_list = np.zeros([len(v_input),len(p_input)])
    for i in enumerate(v.transpose()):
        first_term = v.transpose()[i[0]]-p_star[:,i[0]]
        first_term.reshape(1,2)
        weights = w[:,:,i[0]].reshape(4)
        second_term = np.zeros([2,2])
        for j in enumerate(p_hat[:,:,i[0]]):
            p_j = j[1].reshape(1,2)
            second_term += np.matmul(p_j.transpose(),p_j)*weights[j[0]]
        second_term = np.linalg.inv(second_term)
        for k in enumerate(p_hat[:,:,i[0]]):
            weighted_p_k = np.transpose(k[1].reshape(1, 2)*weights[k[0]])

            A_j = np.matmul(np.matmul(first_term,second_term),weighted_p_k)
            A_list[i[0],k[0]] = A_j

    ## Calculation done for p, calculate new values of v
    new_v_list = []
    for i in enumerate(v.transpose()):
        A_vector = A_list[i[0], :].reshape(4,1)
        new_v = np.sum(q_hat[:,:,i[0]]*A_vector,axis=0).reshape(1,2)+q_star[:,i[0]].reshape(1,2)
        new_v_list.append(new_v.reshape(2))
    return np.array(new_v_list)


ptest = [[1,1], [1,2], [2,1], [2,2]]
qtest = [[0.8,0.8], [1.2,1.8], [2.2,0.8], [2.2,2.2]]
vtest = [[0.5,0.5], [0.5,1.5], [0.5,2.5], [1.5,0.5], [1.5,1.5], [1.5,2.5], [2.5,0.5], [2.5,1.5], [2.5,2.5]]
np_v = np.array(vtest)
new_v = (mls_affine_transform(ptest,qtest,vtest,1))

plt.scatter(new_v[:,0],new_v[:,1])
plt.scatter(np_v[:,0],np_v[:,1])
plt.scatter(np.array(ptest)[:,0],np.array(ptest)[:,1])
plt.scatter(np.array(qtest)[:,0],np.array(qtest)[:,1])

plt.show()