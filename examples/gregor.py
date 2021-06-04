import pymcao

mcao = pymcao.Simulator('gregor_scao.ini')
mcao.init_simulation()
mcao.init_time()

# if (mcao.operation_mode == 'mcao'):
#     for i in range(200):        
#         mcao.frame_mcao(silence=False)            

# if (mcao.operation_mode == 'scao'):        
#     for i in range(200):
#         mcao.frame_scao(silence=True)

# if (mcao.operation_mode == 'mcao_single'):
#     mcao.frame_mcao(silence=False, plot=True)

# if (mcao.operation_mode == 'scao_single'):

mcao.frame_scao(silence=False, plot=True)

mcao.end_time()
mcao.print_time()
mcao.finalize()