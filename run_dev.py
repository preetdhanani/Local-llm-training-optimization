import subprocess
import sys
import os
import threading
import time

def log_reader(pipe, prefix):
    """Reads lines from the pipe and prints them with a prefix."""
    try:
        with pipe:
            for line in iter(pipe.readline, ''):
                sys.stdout.write(f"{prefix} {line}")
                sys.stdout.flush()
    except Exception:
        pass

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_dir = os.path.join(root_dir, "dashboard")
    node_modules_dir = os.path.join(dashboard_dir, "node_modules")
    
    # 1. Ensure frontend dependencies are installed
    if not os.path.exists(node_modules_dir):
        print("[System] Dashboard node_modules not found. Running 'npm install' in dashboard/...")
        try:
            subprocess.run("npm install", shell=True, cwd=dashboard_dir, check=True)
            print("[System] npm install completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[System] Error running npm install: {e}", file=sys.stderr)
            sys.exit(1)
            
    # 2. Spawn backend (FastAPI)
    print("[System] Starting FastAPI backend on http://127.0.0.1:8000...")
    backend_cmd = [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]
    
    backend_process = subprocess.Popen(
        backend_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=root_dir
    )
    
    # 3. Spawn frontend (React/Vite)
    print("[System] Starting React/Vite dashboard on http://localhost:6767...")
    frontend_process = subprocess.Popen(
        "npm run dev -- --port 6767",
        shell=True,
        cwd=dashboard_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # 4. Start threads to read output streams asynchronously
    t_backend = threading.Thread(target=log_reader, args=(backend_process.stdout, "[Backend]"))
    t_frontend = threading.Thread(target=log_reader, args=(frontend_process.stdout, "[Frontend]"))
    
    t_backend.daemon = True
    t_frontend.daemon = True
    
    t_backend.start()
    t_frontend.start()
    
    print("[System] Both servers started. Press Ctrl+C to stop them.")
    
    # 5. Monitor processes and wait for shutdown signal
    try:
        while True:
            backend_code = backend_process.poll()
            frontend_code = frontend_process.poll()
            
            if backend_code is not None:
                print(f"\n[System] Backend exited with code {backend_code}")
                break
            if frontend_code is not None:
                print(f"\n[System] Frontend exited with code {frontend_code}")
                break
                
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[System] Shutting down both servers...")
    finally:
        # 6. Clean up both processes and their children/grandchildren
        if os.name == 'nt':
            # On Windows, terminating a shell subprocess (like npm) leaves orphaned nodes.
            # Using taskkill /F /T kills the process tree starting from PID.
            for proc, name in [(backend_process, "Backend"), (frontend_process, "Frontend")]:
                if proc.poll() is None:
                    try:
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except Exception:
                        proc.terminate()
        else:
            # On Unix-like systems
            for proc in (backend_process, frontend_process):
                if proc.poll() is None:
                    try:
                        proc.terminate()
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        
        print("[System] Clean exit completed.")

if __name__ == "__main__":
    main()
