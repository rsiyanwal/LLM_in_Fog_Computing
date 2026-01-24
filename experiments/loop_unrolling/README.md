# Command to use
```
perf record -F 999 -g -- ~/edge-llm-bench/bin/main_blas -m ~/edge-llm-bench/models/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf -p "hello, how are you?" -n 64
ctrl + c
perf report
```
Fish out the top busy kernels. For example, in my output:
```
+   93.99%     0.04%  main_blas  libggml-cpu.so.0.9.5   [.] ggml_graph_compute_thread.isra.0
+   92.79%     0.00%  main_blas  libc.so.6              [.] 0x0000007fb727de8c
+   92.79%     0.00%  main_blas  libc.so.6              [.] 0x0000007fb7215f78
+   92.05%     0.49%  main_blas  libggml-cpu.so.0.9.5   [.] ggml_compute_forward_mul_mat
+   74.47%    74.37%  main_blas  libggml-cpu.so.0.9.5   [.] ggml_vec_dot_q5_K_q8_K
+   70.41%     0.00%  main_blas  libgomp.so.1.0.0       [.] 0x0000007fb6eaec6c
+   23.63%     0.00%  main_blas  libllama.so.0.0.7816   [.] llama_decode
+   23.63%     0.00%  main_blas  libllama.so.0.0.7816   [.] llama_context::decode(llama_batch const&)
+   23.61%     0.00%  main_blas  libllama.so.0.0.7816   [.] llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, ll
+   23.60%     0.00%  main_blas  libllama.so.0.0.7816   [.] llama_context::graph_compute(ggml_cgraph*, bool)
+   23.60%     0.00%  main_blas  libggml-base.so.0.9.5  [.] ggml_backend_sched_graph_compute_async
+   23.60%     0.00%  main_blas  libggml-cpu.so.0.9.5   [.] ggml_backend_cpu_graph_compute(ggml_backend*, ggml_cgraph*)
+   23.59%     0.00%  main_blas  libggml-cpu.so.0.9.5   [.] ggml_graph_compute
+   23.59%     0.00%  main_blas  libgomp.so.1.0.0       [.] GOMP_parallel
+   22.38%     0.00%  main_blas  libstdc++.so.6.0.33    [.] 0x0000007fb751b4e0
+   22.38%     0.00%  main_blas  llama_blas             [.] server_queue::start_loop(long)
+   22.38%     0.00%  main_blas  llama_blas             [.] server_context_impl::update_slots()
+   16.10%    16.09%  main_blas  libggml-cpu.so.0.9.5   [.] ggml_vec_dot_q6_K_q8_K
+    7.13%     0.00%  main_blas  llama_blas             [.] _start
+    7.13%     0.00%  main_blas  libc.so.6              [.] __libc_start_main
+    7.13%     0.00%  main_blas  libc.so.6              [.] 0x0000007fb71b229c
+    7.13%     0.00%  main_blas  llama_blas             [.] main
+    7.07%     0.00%  main_blas  llama_blas             [.] server_context_impl::load_model(common_params const&)
+    7.07%     0.00%  main_blas  llama_blas             [.] common_init_from_params(common_params&)
+    5.73%     0.00%  main_blas  llama_blas             [.] common_init_result::common_init_result(common_params&)
+    5.45%     0.00%  main_blas  libllama.so.0.0.7816   [.] llama_model_load_from_file
+    5.45%     0.00%  main_blas  libllama.so.0.0.7816   [.] llama_model_load_from_file_impl(std::__cxx11::basic_string<char, std:
+    3.98%     0.00%  main_blas  libllama.so.0.0.7816   [.] llama_model::load_tensors(llama_model_loader&)
+    3.92%     0.00%  main_blas  libllama.so.0.0.7816   [.] llama_model_loader::load_all_data(ggml_context*, std::unordered_map<u
+    3.90%     0.00%  main_blas  libllama.so.0.0.7816   [.] llama_file::read_raw(void*, unsigned long)
+    3.62%     0.00%  main_blas  [kernel.kallsyms]      [k] el0t_64_sync
+    3.62%     0.00%  main_blas  [kernel.kallsyms]      [k] el0t_64_sync_handler
+    2.06%     0.00%  main_blas  [kernel.kallsyms]      [k] handle_mm_fault

```
~74% time was spent on `ggml_vec_dot_q5_K_q8_K` and ~16% on `ggml_vec_dot_q6_K_q8_K`. Even within them:
```
Samples: 21K of event 'cycles:P', Event count (approx.): 37813605236
  Children      Self  Command    Shared Object          Symbol
+   93.99%     0.04%  main_blas  libggml-cpu.so.0.9.5   [.] ggml_graph_compute_thread.isra.0                                    ◆
+   92.79%     0.00%  main_blas  libc.so.6              [.] 0x0000007fb727de8c                                                  ▒
+   92.79%     0.00%  main_blas  libc.so.6              [.] 0x0000007fb7215f78                                                  ▒
+   92.05%     0.49%  main_blas  libggml-cpu.so.0.9.5   [.] ggml_compute_forward_mul_mat                                        ▒
-   74.47%    74.37%  main_blas  libggml-cpu.so.0.9.5   [.] ggml_vec_dot_q5_K_q8_K                                              ▒
   - 73.51% 0x7fb727de8c                                                                                                        ▒
      - 0x7fb7215f78                                                                                                            ▒
         - 55.85% 0x7fb6eaec6c                                                                                                  ▒
              ggml_graph_compute_thread.isra.0                                                                                  ▒
              ggml_compute_forward_mul_mat                                                                                      ▒
              ggml_vec_dot_q5_K_q8_K                                                                                            ▒
         - 17.66% 0x7fb751b4e0                                                                                                  ▒
              server_queue::start_loop(long)                                                                                    ▒
              server_context_impl::update_slots()                                                                               ▒
              llama_decode                                                                                                      ▒
              llama_context::decode(llama_batch const&)                                                                         ▒
              llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&)         ▒
              llama_context::graph_compute(ggml_cgraph*, bool)                                                                  ▒
              ggml_backend_sched_graph_compute_async                                                                            ▒
              ggml_backend_cpu_graph_compute(ggml_backend*, ggml_cgraph*)                                                       ▒
              ggml_graph_compute                                                                                                ▒
              GOMP_parallel                                                                                                     ▒
              ggml_graph_compute_thread.isra.0                                                                                  ▒
              ggml_compute_forward_mul_mat                                                                                      ▒
              ggml_vec_dot_q5_K_q8_K                                                                                            ▒
   + 0.86% _start                                                                                                               ▒
+   70.41%     0.00%  main_blas  libgomp.so.1.0.0       [.] 0x0000007fb6eaec6c                                                  ▒
+   23.63%     0.00%  main_blas  libllama.so.0.0.7816   [.] llama_decode                                                        ▒
+   23.63%     0.00%  main_blas  libllama.so.0.0.7816   [.] llama_context::decode(llama_batch const&)                           ▒
+   23.61%     0.00%  main_blas  libllama.so.0.0.7816   [.] llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, l▒
+   23.60%     0.00%  main_blas  libllama.so.0.0.7816   [.] llama_context::graph_compute(ggml_cgraph*, bool)                    ▒
+   23.60%     0.00%  main_blas  libggml-base.so.0.9.5  [.] ggml_backend_sched_graph_compute_async
+   23.60%     0.00%  main_blas  libggml-cpu.so.0.9.5   [.] ggml_backend_cpu_graph_compute(ggml_backend*, ggml_cgraph*)         ▒
+   23.59%     0.00%  main_blas  libggml-cpu.so.0.9.5   [.] ggml_graph_compute                                                  ▒
+   23.59%     0.00%  main_blas  libgomp.so.1.0.0       [.] GOMP_parallel                                                       ▒
+   22.38%     0.00%  main_blas  libstdc++.so.6.0.33    [.] 0x0000007fb751b4e0                                                  ▒
+   22.38%     0.00%  main_blas  llama_blas             [.] server_queue::start_loop(long)                                      ▒
+   22.38%     0.00%  main_blas  llama_blas             [.] server_context_impl::update_slots()                                 ▒
-   16.10%    16.09%  main_blas  libggml-cpu.so.0.9.5   [.] ggml_vec_dot_q6_K_q8_K                                              ▒
     15.88% 0x7fb727de8c                                                                                                        ▒
      - 0x7fb7215f78                                                                                                            ▒
         - 12.07% 0x7fb6eaec6c                                                                                                  ▒
              ggml_graph_compute_thread.isra.0                                                                                  ▒
              ggml_compute_forward_mul_mat                                                                                      ▒
              ggml_vec_dot_q6_K_q8_K                                                                                            ▒
         - 3.81% 0x7fb751b4e0                                                                                                   ▒
              server_queue::start_loop(long)                                                                                    ▒
              server_context_impl::update_slots()                                                                               ▒
              llama_decode                                                                                                      ▒
              llama_context::decode(llama_batch const&)                                                                         ▒
              llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&)         ▒
              llama_context::graph_compute(ggml_cgraph*, bool)                                                                  ▒
              ggml_backend_sched_graph_compute_async                                                                            ▒
              ggml_backend_cpu_graph_compute(ggml_backend*, ggml_cgraph*)                                                       ▒
              ggml_graph_compute                                                                                                ▒
              GOMP_parallel                                                                                                     ▒
              ggml_graph_compute_thread.isra.0                                                                                  ▒
              ggml_compute_forward_mul_mat                                                                                      ▒
              ggml_vec_dot_q6_K_q8_K                                                                                            ▒
+    7.13%     0.00%  main_blas  llama_blas             [.] _start                                                              ▒
+    7.13%     0.00%  main_blas  libc.so.6              [.] __libc_start_main                                                   ▒
+    7.13%     0.00%  main_blas  libc.so.6              [.] 0x0000007fb71b229c                                                  ▒
```
