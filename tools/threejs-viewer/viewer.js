// A stock three.js glTF viewer: nothing here is ours but this glue. three.js draws the model (GLTFLoader, MeshStandardMaterial PBR, PMREM image-based lighting from the
// bundled RoomEnvironment, ACES tone mapping, a shadow-casting key light). Used to see what a renderer we did not write makes of our GLBs.
//   index.html?glb=<url>[&yaw=<deg>][&pitch=<deg>][&shot=1][&size=<px>]      shot=1: no UI, no auto-rotate, window.__ready = true once the model is drawn
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const q = new URLSearchParams(location.search);
const shot = q.has('shot');
if (shot) document.body.classList.add('shot');

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: shot });
renderer.setPixelRatio(shot ? 1 : Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f141f);
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

const camera = new THREE.PerspectiveCamera(35, innerWidth / innerHeight, 0.01, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const key = new THREE.DirectionalLight(0xffffff, 2.2);
key.castShadow = true; key.shadow.mapSize.set(2048, 2048); key.shadow.bias = -0.0004; key.shadow.normalBias = 0.01;
scene.add(key);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(20, 20), new THREE.ShadowMaterial({ opacity: 0.35 }));
floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);

let current = null, stats = '';
const loader = new GLTFLoader();

function frame(object, yawDeg, pitchDeg) {
  const box = new THREE.Box3().setFromObject(object);
  const size = box.getSize(new THREE.Vector3()), center = box.getCenter(new THREE.Vector3());
  object.position.sub(center); object.position.y += size.y / 2;           // sit on the floor, centred
  floor.position.y = 0;
  const radius = Math.max(size.x, size.y, size.z) * 0.5;
  const dist = radius / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.25;
  const yaw = THREE.MathUtils.degToRad(yawDeg), pitch = THREE.MathUtils.degToRad(pitchDeg);
  const target = new THREE.Vector3(0, size.y / 2, 0);
  camera.position.set(target.x + dist * Math.sin(yaw) * Math.cos(pitch), target.y + dist * Math.sin(pitch), target.z + dist * Math.cos(yaw) * Math.cos(pitch));
  controls.target.copy(target); controls.update();
  key.position.set(radius * 2.2, radius * 3.2, radius * 2.0); key.target.position.copy(target); scene.add(key.target);
  const s = radius * 2.5; Object.assign(key.shadow.camera, { left: -s, right: s, top: s, bottom: -s, near: 0.1, far: radius * 12 }); key.shadow.camera.updateProjectionMatrix();
}

async function load(url) {
  const gltf = await loader.loadAsync(url);
  if (current) scene.remove(current);
  current = gltf.scene;
  let tris = 0, meshes = 0;
  current.traverse(o => { if (o.isMesh) { meshes++; o.castShadow = true; o.receiveShadow = true; const g = o.geometry; tris += (g.index ? g.index.count : g.attributes.position.count) / 3; } });
  scene.add(current);
  frame(current, Number(q.get('yaw') ?? 20), Number(q.get('pitch') ?? 8));
  stats = `${meshes} mesh(es), ${Math.round(tris).toLocaleString()} triangles`;
  document.getElementById('info').textContent += '  |  ' + stats;
  return gltf;
}

let rotating = false;
document.getElementById('rotate').onchange = e => { rotating = e.target.checked; controls.autoRotate = rotating; };
document.getElementById('wire').onchange = e => current?.traverse(o => { if (o.isMesh) { const ms = Array.isArray(o.material) ? o.material : [o.material]; ms.forEach(m => m.wireframe = e.target.checked); } });
document.getElementById('exposure').oninput = e => renderer.toneMappingExposure = Number(e.target.value);
document.getElementById('file').onchange = e => { const f = e.target.files[0]; if (f) load(URL.createObjectURL(f)); };
const list = document.getElementById('list');
fetch('models.json').then(r => r.ok ? r.json() : []).catch(() => []).then(models => {
  for (const m of models) list.add(new Option(m.name, m.url));
  list.onchange = () => load(list.value);
  const start = q.get('glb') || models[0]?.url;
  if (start) { if (!models.some(m => m.url === start)) list.add(new Option(start, start, true, true)); else list.value = start; load(start).then(() => { requestAnimationFrame(() => requestAnimationFrame(() => { window.__ready = true; })); }); }
});

addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
