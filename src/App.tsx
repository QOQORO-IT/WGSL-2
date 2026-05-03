import { useEffect, useRef } from "react";
import { VoxelMeshRenderer, setupInput, updateCamera, camera } from "./voxel-mesh";

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<VoxelMeshRenderer | null>(null);
  const animRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const renderer = new VoxelMeshRenderer();
    rendererRef.current = renderer;

    let lastTime = performance.now();
    let frameCount = 0;
    let fps = 0;
    const fpsElem = document.getElementById("fps");

    async function init() {
      try {
        if (!canvas) return;
        await renderer.init(canvas);
        if (!canvas) return;
        setupInput(canvas);

        function frame() {
          const now = performance.now();
          const deltaTime = Math.min((now - lastTime) / 1000, 0.1);
          lastTime = now;

          updateCamera(deltaTime);
          renderer.render();

          frameCount++;
          if (now % 1000 < 20 && fpsElem) {
            fps = Math.round(frameCount / (deltaTime || 0.016));
            frameCount = 0;
            fpsElem.textContent = `FPS: ${fps} | Quads: ~${(renderer.indexCount / 6).toLocaleString()} | Position: ${camera.position.map((v: number) => v.toFixed(1)).join(", ")}`;
          }

          animRef.current = requestAnimationFrame(frame);
        }

        animRef.current = requestAnimationFrame(frame);
      } catch (err: any) {
        console.error("Failed to initialize WebGPU:", err);
        const errorDiv = document.getElementById("error");
        if (errorDiv) {
          errorDiv.textContent = `Error: ${err.message || err}`;
          errorDiv.style.display = "block";
        }
      }
    }

    const handleResize = () => {
      const scale = 1.0;
      canvas.width = Math.floor(window.innerWidth * scale);
      canvas.height = Math.floor(window.innerHeight * scale);
      canvas.style.width = "100%";
      canvas.style.height = "100%";
      renderer.resize(canvas.width, canvas.height);
    };
    handleResize();
    window.addEventListener("resize", handleResize);

    init();

    return () => {
      cancelAnimationFrame(animRef.current);
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-black">
      <canvas
        ref={canvasRef}
        className="block w-full h-full"
        style={{ imageRendering: "auto" }}
      />
      <div className="absolute top-4 left-4 text-white font-mono text-sm bg-black/60 px-3 py-2 rounded select-none pointer-events-none">
        <div id="fps">Initializing mesh renderer...</div>
        <div className="text-xs text-gray-400 mt-1">
          WASD = Move | Space = Up | Shift = Down | Mouse = Look | Click = Lock Cursor | Esc = Unlock
        </div>
      </div>
      <div
        id="error"
        className="absolute inset-0 flex items-center justify-center bg-black/80 text-red-400 text-center p-8 hidden"
      >
        <div>
          <h2 className="text-xl font-bold mb-2">WebGPU Error</h2>
          <p>Your browser does not support WebGPU, or it is not enabled.</p>
          <p className="text-sm mt-2 text-gray-400">
            Try Chrome 113+ or Edge 113+ with WebGPU enabled in flags.
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;