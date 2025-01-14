import multiprocessing as mp
import time

def mock_disk_prefetch_worker(queue, cache, stop, lock):
    """
    Simplified worker process to fetch items and add them to the shared cache.
    """
    try:
        while not stop.value:
            layer_idx = queue.get(timeout=0.5)
            if layer_idx is None:
                print("Worker received termination signal.")
                break
            print(f"Worker fetched layer {layer_idx}")
            with lock:
                if layer_idx not in cache:
                    cache[layer_idx] = f"Layer_{layer_idx}"  # Mock content
                    print(f"Worker added layer {layer_idx} to cache.")
    except Exception as e:
        print(f"Worker encountered an error: {e}")

def simulate_inference(num_layers, queue, cache, lock, initial_cache, schedule):
    """
    Simulate layer-by-layer inference, adding layers dynamically to the queue.
    """
    for i in range(initial_cache):
        with lock:
            if i not in cache and i not in schedule:
                schedule.add(i)
                queue.put(i)
                print(f"Scheduled initial layer {i}")

    for current_layer in range(num_layers):
        # Wait for the layer to be available in the cache
        while True:
            with lock:
                if current_layer in cache:
                    break
            time.sleep(0.1)  # Avoid busy waiting

        # Process the layer
        with lock:
            print(f"Processing layer {current_layer} (content: {cache[current_layer]})")

        # Schedule the next layer
        next_layer = current_layer + 1
        if next_layer < num_layers:
            with lock:
                if next_layer not in cache and next_layer not in schedule:
                    schedule.add(next_layer)
                    queue.put(next_layer)
                    print(f"Scheduled next layer {next_layer}")

if __name__ == "__main__":
    # Local variables for the test
    prefetch_queue = mp.Queue()
    stop_signal = mp.Value('b', False)
    manager = mp.Manager()
    layer_weights_cache = manager.dict()  # Shared dictionary for cache
    scheduled_layers = set()
    cache_lock = mp.Lock()
    INITIAL_CACHED_LAYERS = 10

    # Start the worker process
    num_layers = 15  # Number of layers to process
    worker_process = mp.Process(
        target=mock_disk_prefetch_worker,
        args=(prefetch_queue, layer_weights_cache, stop_signal, cache_lock),
    )
    worker_process.start()

    # Run the inference simulation
    simulate_inference(
        num_layers=num_layers,
        queue=prefetch_queue,
        cache=layer_weights_cache,
        lock=cache_lock,
        initial_cache=INITIAL_CACHED_LAYERS,
        schedule=scheduled_layers,
    )

    # Stop the worker
    stop_signal.value = True
    prefetch_queue.put(None)
    worker_process.join()

    # Check the final cache content
    print("Final cache content:", dict(layer_weights_cache))
    print("Test completed with shared cache and proper multiprocessing.")
