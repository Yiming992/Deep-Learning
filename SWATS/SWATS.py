from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.training import optimizer

class SWATS(optimizer.Optimizer):

    def __init__(self,adam_lr=1e-3,beta1=0.9,beta2=0.999,epsilon=1e-9,use_locking=False,name='SWATS'):
        super(SWATS,self).__init__(use_locking,name)
        self._adam_lr=adam_lr
        self._beta1=beta1
        self._beta2=beta2
        self._epsilon=epsilon
        self._Lambda=None
        self._gamma=None
        self._switch=None
        self._step=None

        self._adam_lr_t=None
        self._beta1_t=None
        self._beta2_t=None
        self._epsilon_t=None

    def _create_slots(self,var_list):
        """
        Create additional variables and slots for the optimizer
        """
        self._step=variable_scope.variable(initial_value=0,name='step',trainable=False)
        self._switch=variable_scope.variable(initial_value=0.0,name='switch',trainable=False)
        self._gamma=variable_scope.variable(initial_value=0.0,name='gamma',trainable=False)
        self._Lambda=variable_scope.variable(initial_value=0.0,name='lambda',trainable=False)
        for v in var_list:
            self._zeros_slot(v,'mk',self._name)
            self._zeros_slot(v,'ak',self._name)
            self._zeros_slot(v,'vk',self._name)

    def _prepare(self):
        self._adam_lr_t=ops.convert_to_tensor(self._adam_lr,name='adam_lr')
        self._beta1_t=ops.convert_to_tensor(self._beta1,name='beta1')
        self._beta2_t=ops.convert_to_tensor(self._beta2,name='beta2')
        self._epsilon_t=ops.convert_to_tensor(self._epsilon,name='epsilon')

    def _apply_dense(self,grad,var):
        """
        apply graidents densely
        """
        graph=ops.get_default_graph()
        #step=self._step#math_ops.cast(self._step,var.dtype.base_dtype)
        #switch=self._switch#math_ops.cast(self._switch,var.dtype.base_dtype)
        #Lambda=self._Lambda#math_ops.cast(self._Lambda,var.dtype.base_dtype)
        #gamma=self._gamma#math_ops.cast(self._gamma,var.dtype.base_dtype)
        adam_lr_t=math_ops.cast(self._adam_lr_t,var.dtype.base_dtype)
        beta1_t=math_ops.cast(self._beta1_t,var.dtype.base_dtype)
        beta2_t=math_ops.cast(self._beta2_t,var.dtype.base_dtype)
        epsilon_t=math_ops.cast(self._epsilon_t,var.dtype.base_dtype)

        step=state_ops.assign(self._step,self._step+1)
        lr=adam_lr_t*math_ops.sqrt(1-math_ops.pow(beta2_t,step))/(1-math_ops.pow(beta1_t,step))

        def is_true():
            vk=self.get_slot(var,'vk')
            vk_t=state_ops.assign(vk,beta1_t*vk+grad)
            var_update=state_ops.assign_sub(var,(1-beta1_t)*self._gamma*vk)
            return control_flow_ops.group(*[var_update,vk_t])

        def is_false():
            mk=self.get_slot(var,'mk')
            mk_t=state_ops.assign(mk,beta1_t*mk+(1-beta2_t)*grad)
            ak=self.get_slot(var,'ak')
            ak_t=state_ops.assign(ak,beta2_t*ak+(1-beta2_t)*(grad*grad))
            pk=-lr*mk_t/(math_ops.sqrt(ak_t)+epsilon_t)
            var_update=state_ops.assign_sub(var,-pk)
            condition=math_ops.reduce_sum(math_ops.multiply(pk,grad))
            def is_true2():
                yk=math_ops.reduce_sum(math_ops.multiply(pk,grad))/-math_ops.reduce_sum(math_ops.multiply(pk,grad))
                Lambda_t=state_ops.assign(self._Lambda,beta2_t*self._Lambda+(1-beta2_t)*yk)
                def is_true3():
                    switch=state_ops.assign(self._switch,1)
                    gamma=state_ops.assign(self._gamma,Lambda_t/(1-beta2_t**step))
                    return switch,gamma
                def is_false3():
                    switch=self._switch
                    gamma=self._gamma
                    return switch,gamma
                switch,gamma=control_flow_ops.cond(gen_math_ops.less(math_ops.abs(Lambda_t/(1-beta2_t**step)-self._gamma),epsilon_t),is_true3,is_false3)
                return control_flow_ops.group(*[var_update,mk,ak])

            return control_flow_ops.group(*[var_update,mk,ak])

            def is_false2():
                return control_flow_ops.group(*[var_update,mk,ak])
            ops_group=control_flow_ops.cond(gen_math_ops.not_equal(condition,0),is_true2,is_false2)
        ops_group=control_flow_ops.cond(gen_math_ops.equal(self._switch,1),is_true,is_false)

        return ops_group
