<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue';
import Chart from 'primevue/chart';
import Button from 'primevue/button';
import Avatar from 'primevue/avatar';
import ProgressBar from 'primevue/progressbar';
import Tag from 'primevue/tag';
import FileUpload from 'primevue/fileupload';
import SelectButton from 'primevue/selectbutton';
import Message from 'primevue/message';

const activeTab = ref('dashboard');

// --- Dashboard Live Data ---
const chartData = ref(null);
const chartOptions = ref(null);
let timeIndex = 0;
const generateECGDataPoint = () => {
  const t = timeIndex % 100;
  timeIndex++;
  if (t > 10 && t < 20) return 0.2 * Math.sin((t - 10) * Math.PI / 10);
  if (t === 30) return -0.3;
  if (t === 32) return 1.5;
  if (t === 34) return -0.4;
  if (t > 50 && t < 70) return 0.4 * Math.sin((t - 50) * Math.PI / 20);
  return (Math.random() - 0.5) * 0.05;
};
const dataPoints = Array.from({ length: 150 }, () => 0);
let intervalId;
const updateChart = () => {
  dataPoints.push(generateECGDataPoint());
  dataPoints.shift();
  chartData.value = {
    labels: Array.from({ length: 150 }, (_, i) => i),
    datasets: [{
      label: 'Lead II',
      data: [...dataPoints],
      borderColor: '#3b82f6',
      tension: 0.4,
      borderWidth: 2.5,
      pointRadius: 0,
    }]
  };
};

// --- Model 1: Signal -> Class ---
const signalPresets = [
    { label: 'Normal', value: 'N', color: '#10b981' },
    { label: 'PVC', value: 'V', color: '#ef4444' },
    { label: 'Atrial Fib', value: 'S', color: '#f59e0b' }
];
const selectedSignal = ref('N');
const signalClassData = ref(null);
const isClassifyingSignal = ref(false);
const signalResults = ref(null);

const generateSignal = (type) => {
    const arr = [];
    for(let t=0; t<100; t++) {
        if(type === 'N') {
            if (t > 10 && t < 20) arr.push(0.2 * Math.sin((t - 10) * Math.PI / 10));
            else if (t === 30) arr.push(-0.3); else if (t === 32) arr.push(1.5); else if (t === 34) arr.push(-0.4);
            else if (t > 50 && t < 70) arr.push(0.4 * Math.sin((t - 50) * Math.PI / 20));
            else arr.push((Math.random()-0.5)*0.02);
        } else if (type === 'V') {
             if (t > 25 && t < 45) arr.push(1.8 * Math.sin((t - 25) * Math.PI / 20));
             else if (t > 50 && t < 80) arr.push(-0.6 * Math.sin((t - 50) * Math.PI / 30));
             else arr.push((Math.random()-0.5)*0.02);
        } else {
             arr.push(0.5 * Math.sin(t * 0.5) + (Math.random()-0.5)*0.3);
        }
    }
    return arr;
};

const loadSignalData = () => {
    signalClassData.value = {
        labels: Array.from({ length: 100 }, (_, i) => i),
        datasets: [{
            label: '1D Signal Input',
            data: generateSignal(selectedSignal.value),
            borderColor: signalPresets.find(p => p.value === selectedSignal.value).color,
            tension: 0.4, borderWidth: 3, pointRadius: 0
        }]
    };
    signalResults.value = null;
};

const runSignalInference = () => {
    isClassifyingSignal.value = true;
    setTimeout(() => {
        isClassifyingSignal.value = false;
        if(selectedSignal.value === 'N') signalResults.value = { class: 'Normal (N)', conf: 98.2 };
        else if(selectedSignal.value === 'V') signalResults.value = { class: 'Ventricular (V)', conf: 94.5 };
        else signalResults.value = { class: 'Supraventricular (S)', conf: 89.1 };
    }, 1500);
};

// --- Model 2: Image -> Signal ---
const extractionImage = ref(null);
const isExtracting = ref(false);
const extractedSignal = ref(null);
const extractionStep = ref(0);

const onImageSelect = (event) => {
    const file = event.files[0];
    const reader = new FileReader();
    reader.onload = (e) => {
        extractionImage.value = e.target.result;
        extractedSignal.value = null;
    };
    reader.readAsDataURL(file);
};

const runExtraction = () => {
    isExtracting.value = true;
    extractionStep.value = 1;
    setTimeout(() => extractionStep.value = 2, 1000);
    setTimeout(() => extractionStep.value = 3, 2000);
    setTimeout(() => {
        isExtracting.value = false;
        extractedSignal.value = {
            labels: Array.from({ length: 200 }, (_, i) => i),
            datasets: [{
                label: 'Reconstructed Waveform',
                data: Array.from({ length: 200 }, () => generateECGDataPoint()),
                borderColor: '#8b5cf6',
                tension: 0.3, borderWidth: 2, pointRadius: 0
            }]
        };
    }, 3500);
};

// --- Model 3: Image -> Class (E2E) ---
const e2eImage = ref(null);
const e2eStatus = ref('idle'); // idle, processing, done
const e2eResults = ref(null);

const onE2ESelect = (event) => {
    e2eImage.value = URL.createObjectURL(event.files[0]);
    e2eStatus.value = 'idle';
};

const runE2E = () => {
    e2eStatus.value = 'processing';
    setTimeout(() => {
        e2eStatus.value = 'done';
        e2eResults.value = {
            diagnosis: 'Premature Ventricular Contraction',
            confidence: 96.8,
            details: 'The end-to-end model identified a wide QRS complex directly from the image features with high spatial attention on the mid-strip region.'
        };
    }, 4000);
};

// Shared Options
const baseOptions = {
    animation: true, responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: { x: { display: false }, y: { grid: { color: 'rgba(0,0,0,0.05)' } } }
};

onMounted(() => {
  chartOptions.value = { ...baseOptions, animation: false };
  updateChart();
  intervalId = setInterval(updateChart, 30);
  loadSignalData();
});

onUnmounted(() => clearInterval(intervalId));
</script>

<template>
  <div class="min-h-screen bg-slate-50 text-slate-800 flex overflow-hidden font-sans">
    <!-- Sidebar -->
    <aside class="w-20 lg:w-72 bg-white border-r border-slate-200 flex flex-col justify-between hidden md:flex z-20 shadow-[4px_0_24px_rgba(0,0,0,0.02)]">
      <div class="p-6">
        <div class="flex items-center gap-3 mb-10 px-2">
          <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-700 flex items-center justify-center shadow-lg shadow-blue-500/30">
            <i class="pi pi-bolt text-white text-xl"></i>
          </div>
          <h1 class="text-xl font-bold text-slate-900 hidden lg:block tracking-tight">ECG Inspire <span class="text-blue-600">AI</span></h1>
        </div>
        
        <nav class="space-y-1">
          <button @click="activeTab = 'dashboard'" :class="['w-full flex items-center gap-3 p-3.5 rounded-xl transition-all text-left group', activeTab === 'dashboard' ? 'bg-blue-50 text-blue-700 font-bold' : 'text-slate-500 hover:bg-slate-50']">
            <i class="pi pi-th-large text-lg" :class="activeTab === 'dashboard' ? 'text-blue-600' : 'group-hover:text-slate-700'"></i>
            <span class="hidden lg:block">Overview</span>
          </button>
          
          <div class="pt-4 pb-2 px-4 hidden lg:block">
            <span class="text-[10px] font-bold text-slate-400 uppercase tracking-widest">AI Demos</span>
          </div>

          <button @click="activeTab = 'model1'" :class="['w-full flex items-center gap-3 p-3.5 rounded-xl transition-all text-left group', activeTab === 'model1' ? 'bg-blue-50 text-blue-700 font-bold' : 'text-slate-500 hover:bg-slate-50']">
            <i class="pi pi-chart-line text-lg" :class="activeTab === 'model1' ? 'text-blue-600' : 'group-hover:text-slate-700'"></i>
            <span class="hidden lg:block">Signal &rarr; Class</span>
          </button>

          <button @click="activeTab = 'model2'" :class="['w-full flex items-center gap-3 p-3.5 rounded-xl transition-all text-left group', activeTab === 'model2' ? 'bg-blue-50 text-blue-700 font-bold' : 'text-slate-500 hover:bg-slate-50']">
            <i class="pi pi-image text-lg" :class="activeTab === 'model2' ? 'text-blue-600' : 'group-hover:text-slate-700'"></i>
            <span class="hidden lg:block">Image &rarr; Signal</span>
          </button>

          <button @click="activeTab = 'model3'" :class="['w-full flex items-center gap-3 p-3.5 rounded-xl transition-all text-left group', activeTab === 'model3' ? 'bg-blue-50 text-blue-700 font-bold' : 'text-slate-500 hover:bg-slate-50']">
            <i class="pi pi-shield text-lg" :class="activeTab === 'model3' ? 'text-blue-600' : 'group-hover:text-slate-700'"></i>
            <span class="hidden lg:block">Image &rarr; Class</span>
          </button>
        </nav>
      </div>
      
      <div class="p-6 border-t border-slate-100 flex items-center gap-3 bg-slate-50/50">
        <Avatar image="https://i.pravatar.cc/150?u=researcher" shape="circle" class="border-2 border-white shadow-sm" />
        <div class="hidden lg:block">
          <p class="text-sm font-bold text-slate-900">Research Sandbox</p>
          <p class="text-[10px] font-medium text-slate-500 uppercase tracking-tight">Model Evaluator</p>
        </div>
      </div>
    </aside>

    <!-- Main Content -->
    <main class="flex-1 overflow-y-auto relative">
      <div class="absolute top-0 left-0 right-0 h-80 bg-gradient-to-b from-blue-50/80 to-transparent pointer-events-none"></div>

      <div class="p-8 relative z-10 max-w-6xl mx-auto">
        
        <!-- Tab: Dashboard -->
        <div v-if="activeTab === 'dashboard'" class="animate-fade-in">
          <header class="mb-10">
            <h2 class="text-4xl font-extrabold tracking-tight text-slate-900 mb-2">System Dashboard</h2>
            <p class="text-lg text-slate-500">Global monitor for ECG signal processing and AI health diagnostics.</p>
          </header>

          <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
             <div class="glass p-6 border-l-4 border-l-blue-500">
                <div class="text-slate-500 text-xs font-bold uppercase tracking-wider mb-2">Active Signals</div>
                <div class="text-3xl font-black text-slate-900">124 <span class="text-sm font-medium text-blue-500 ml-1">Live</span></div>
             </div>
             <div class="glass p-6 border-l-4 border-l-emerald-500">
                <div class="text-slate-500 text-xs font-bold uppercase tracking-wider mb-2">Inference Speed</div>
                <div class="text-3xl font-black text-slate-900">12ms <span class="text-sm font-medium text-emerald-500 ml-1">Avg.</span></div>
             </div>
             <div class="glass p-6 border-l-4 border-l-purple-500">
                <div class="text-slate-500 text-xs font-bold uppercase tracking-wider mb-2">Model Version</div>
                <div class="text-3xl font-black text-slate-900">v3.2.0 <span class="text-sm font-medium text-purple-500 ml-1">Stable</span></div>
             </div>
          </div>

          <div class="glass p-8">
            <div class="flex justify-between items-center mb-6">
                <div>
                    <h3 class="text-xl font-bold text-slate-900">Live ECG Stream</h3>
                    <p class="text-sm text-slate-500">Real-time Lead II visualization</p>
                </div>
                <div class="flex gap-2">
                    <Tag value="Filtered" severity="info" rounded />
                    <Tag value="250Hz" severity="secondary" rounded />
                </div>
            </div>
            <div class="h-64 w-full bg-slate-900/5 rounded-2xl p-4 relative overflow-hidden">
                <!-- Grid -->
                <div class="absolute inset-0 opacity-[0.05]" style="background-image: linear-gradient(#3b82f6 1px, transparent 1px), linear-gradient(90deg, #3b82f6 1px, transparent 1px); background-size: 30px 30px;"></div>
                <Chart type="line" :data="chartData" :options="chartOptions" class="h-full w-full relative z-10" />
            </div>
          </div>
        </div>

        <!-- Tab: Model 1 (Signal -> Class) -->
        <div v-if="activeTab === 'model1'" class="animate-fade-in">
          <header class="mb-10">
            <h2 class="text-4xl font-extrabold tracking-tight text-slate-900 mb-2">1D Signal Classifier</h2>
            <p class="text-lg text-slate-500">Classification of numerical ECG waveforms into cardiac categories.</p>
          </header>

          <div class="grid grid-cols-1 xl:grid-cols-5 gap-8">
            <div class="xl:col-span-3 space-y-6">
                <div class="glass p-8">
                    <div class="flex items-center justify-between mb-8">
                        <h3 class="font-bold text-lg">Input Waveform</h3>
                        <div class="flex gap-2 bg-slate-100 p-1 rounded-xl">
                            <button v-for="p in signalPresets" :key="p.value" @click="selectedSignal = p.value; loadSignalData()" :class="['px-4 py-1.5 rounded-lg text-xs font-bold transition-all', selectedSignal === p.value ? 'bg-white shadow-sm text-slate-900' : 'text-slate-500']">
                                {{ p.label }}
                            </button>
                        </div>
                    </div>
                    <div class="h-64 w-full bg-white rounded-2xl border border-slate-100 p-4 shadow-inner">
                         <Chart type="line" :data="signalClassData" :options="baseOptions" class="h-full w-full" />
                    </div>
                    <div class="mt-8">
                        <Button label="Run Classifier" icon="pi pi-bolt" class="w-full py-4 rounded-2xl text-lg font-bold" @click="runSignalInference" :loading="isClassifyingSignal" />
                    </div>
                </div>
            </div>

            <div class="xl:col-span-2">
                <div class="glass p-8 h-full flex flex-col">
                    <h3 class="font-bold text-lg mb-6">Classification Results</h3>
                    
                    <div v-if="!signalResults && !isClassifyingSignal" class="flex-1 flex flex-col items-center justify-center text-slate-400 text-center">
                        <i class="pi pi-microchip text-5xl mb-4 opacity-20"></i>
                        <p class="text-sm">Select a preset and run the model to see inference results.</p>
                    </div>

                    <div v-else-if="isClassifyingSignal" class="flex-1 flex flex-col items-center justify-center">
                        <div class="w-20 h-20 border-4 border-blue-100 border-t-blue-600 rounded-full animate-spin mb-4"></div>
                        <p class="font-bold text-slate-700">Analyzing Signal...</p>
                        <p class="text-xs text-slate-400 mt-1">Applying 1D-CNN filters</p>
                    </div>

                    <div v-else class="space-y-8 animate-fade-in">
                        <div class="p-6 rounded-2xl bg-blue-50 border border-blue-100 text-center">
                            <div class="text-blue-600 text-xs font-black uppercase tracking-widest mb-1">Primary Prediction</div>
                            <div class="text-3xl font-black text-slate-900">{{ signalResults.class }}</div>
                            <div class="mt-2 inline-flex items-center gap-1.5 px-3 py-1 bg-blue-600 text-white rounded-full text-[10px] font-bold">
                                {{ signalResults.conf }}% Confidence
                            </div>
                        </div>

                        <div class="space-y-4">
                            <h4 class="text-xs font-bold text-slate-500 uppercase">Class Probabilities</h4>
                            <div v-for="p in [ {l:'Normal', v:selectedSignal==='N'?98:2}, {l:'PVC', v:selectedSignal==='V'?94:4}, {l:'AFib', v:selectedSignal==='S'?89:7} ]" :key="p.l">
                                <div class="flex justify-between text-xs font-bold mb-1.5">
                                    <span>{{ p.l }}</span>
                                    <span>{{ p.v }}%</span>
                                </div>
                                <ProgressBar :value="p.v" :showValue="false" style="height: 6px"></ProgressBar>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
          </div>
        </div>

        <!-- Tab: Model 2 (Image -> Signal) -->
        <div v-if="activeTab === 'model2'" class="animate-fade-in">
          <header class="mb-10">
            <h2 class="text-4xl font-extrabold tracking-tight text-slate-900 mb-2">Waveform Reconstructor</h2>
            <p class="text-lg text-slate-500">Extracting 1D heartbeat signals from scanned ECG paper images.</p>
          </header>

          <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div class="glass p-8 overflow-hidden relative group h-[480px] flex flex-col">
                <div v-if="!extractionImage" class="flex-1 border-2 border-dashed border-slate-200 rounded-3xl flex flex-col items-center justify-center p-8 transition-colors hover:border-blue-400">
                    <FileUpload mode="basic" accept="image/*" @select="onImageSelect" customUpload auto chooseLabel="Select ECG Image" class="p-button-lg rounded-2xl" />
                    <p class="text-slate-400 text-sm mt-4">Drop a photo of an ECG strip here</p>
                    <div class="mt-6 flex gap-4">
                        <button @click="extractionImage='/samples/normal.png'; extractedSignal=null" class="text-xs font-bold text-blue-600 hover:underline">Use Sample</button>
                    </div>
                </div>
                <div v-else class="flex-1 flex flex-col items-center justify-center relative min-h-0">
                    <img :src="extractionImage" class="max-w-full max-h-full rounded-2xl shadow-lg object-contain bg-white p-2" />
                    
                    <!-- Scanning Overlay -->
                    <div v-if="isExtracting" class="absolute inset-0 bg-blue-500/10 pointer-events-none overflow-hidden rounded-2xl">
                        <div class="absolute top-0 left-0 w-full h-1 bg-blue-500 shadow-[0_0_20px_rgba(59,130,246,0.8)] animate-scan"></div>
                    </div>

                    <div class="absolute bottom-4 left-0 right-0 flex justify-center gap-2">
                        <Button v-if="!isExtracting && !extractedSignal" label="Begin Extraction" icon="pi pi-expand" class="rounded-xl shadow-lg" @click="runExtraction" />
                        <Button icon="pi pi-times" class="p-button-danger p-button-rounded shadow-lg" @click="extractionImage=null; extractedSignal=null" />
                    </div>
                </div>
            </div>

            <div class="glass p-8 h-[480px] flex flex-col">
                <h3 class="font-bold text-lg mb-6 shrink-0">Extraction Workspace</h3>
                
                <div v-if="isExtracting" class="flex-1 flex flex-col justify-center space-y-6">
                    <div class="p-4 rounded-xl" :class="extractionStep >= 1 ? 'bg-blue-50 border border-blue-100 text-blue-700' : 'text-slate-400'">
                        <i class="pi pi-check-circle mr-2"></i>
                        <span class="text-sm font-bold">1. Grid Detection & Normalization</span>
                    </div>
                    <div class="p-4 rounded-xl" :class="extractionStep >= 2 ? 'bg-blue-50 border border-blue-100 text-blue-700' : 'text-slate-400'">
                        <i :class="extractionStep === 2 ? 'pi pi-spin pi-spinner' : 'pi pi-check-circle'" class="mr-2"></i>
                        <span class="text-sm font-bold">2. Waveform Segmentation</span>
                    </div>
                    <div class="p-4 rounded-xl" :class="extractionStep >= 3 ? 'bg-blue-50 border border-blue-100 text-blue-700' : 'text-slate-400'">
                        <i :class="extractionStep === 3 ? 'pi pi-spin pi-spinner' : 'pi pi-check-circle'" class="mr-2"></i>
                        <span class="text-sm font-bold">3. Interpolation & Mapping</span>
                    </div>
                </div>

                <div v-else-if="extractedSignal" class="animate-fade-in flex-1 flex flex-col min-h-0">
                    <div class="flex-1 bg-slate-900 rounded-2xl p-6 mb-6">
                        <Chart type="line" :data="extractedSignal" :options="baseOptions" class="h-full w-full" />
                    </div>
                    <Message severity="success" icon="pi pi-verified" class="mb-0 shrink-0">
                        Signal successfully digitized. Accuracy: 99.4%
                    </Message>
                </div>

                <div v-else class="flex-1 flex flex-col items-center justify-center text-slate-300">
                    <i class="pi pi-sliders-h text-5xl mb-4"></i>
                    <p class="text-sm">Upload an image to start the reconstruction process.</p>
                </div>
            </div>
          </div>
        </div>

        <!-- Tab: Model 3 (Image -> Class E2E) -->
        <div v-if="activeTab === 'model3'" class="animate-fade-in">
          <header class="mb-10">
            <h2 class="text-4xl font-extrabold tracking-tight text-slate-900 mb-2">E2E Diagnostic Unit</h2>
            <p class="text-lg text-slate-500">End-to-end classification directly from images using Vision Transformers.</p>
          </header>

          <div class="max-w-4xl mx-auto">
            <div class="glass p-10 text-center mb-8">
                <div v-if="!e2eImage">
                    <div class="w-24 h-24 bg-blue-50 rounded-full flex items-center justify-center mx-auto mb-6 text-blue-600">
                        <i class="pi pi-shield text-4xl"></i>
                    </div>
                    <h3 class="text-2xl font-black text-slate-900 mb-4">Upload Diagnostic Strip</h3>
                    <p class="text-slate-500 mb-8 max-w-md mx-auto">Upload a standard clinical ECG strip image. Our end-to-end model will handle feature extraction and diagnosis in one pass.</p>
                    <div class="flex flex-col items-center gap-4">
                        <FileUpload mode="basic" accept="image/*" @select="onE2ESelect" customUpload auto chooseLabel="Choose Image File" class="p-button-lg px-8 rounded-2xl shadow-xl" />
                        <button @click="e2eImage='/samples/anomaly.png'; e2eStatus='idle'" class="text-sm font-bold text-blue-600 hover:underline">Try with Anomaly Sample</button>
                    </div>
                </div>

                <div v-else class="animate-fade-in">
                    <div class="flex flex-col md:flex-row gap-8 items-center">
                        <div class="shrink-0 relative">
                            <img :src="e2eImage" class="w-64 h-64 object-cover rounded-3xl border-4 border-white shadow-2xl shadow-blue-500/20" />
                            <div v-if="e2eStatus === 'processing'" class="absolute inset-0 bg-blue-600/40 rounded-3xl flex items-center justify-center backdrop-blur-[2px]">
                                <i class="pi pi-spin pi-spinner text-white text-3xl"></i>
                            </div>
                        </div>
                        <div class="flex-1 text-left">
                            <div v-if="e2eStatus === 'idle'">
                                <h3 class="text-2xl font-black text-slate-900 mb-2">Image Loaded</h3>
                                <p class="text-slate-500 mb-6">Ready to perform end-to-end neural diagnosis on this sample.</p>
                                <Button label="Start AI Diagnosis" icon="pi pi-play" class="p-button-lg px-10 rounded-2xl" @click="runE2E" />
                            </div>

                            <div v-else-if="e2eStatus === 'processing'">
                                <h3 class="text-2xl font-black text-slate-900 mb-2">Deep Inference...</h3>
                                <p class="text-slate-500 mb-4">Passing through Vision Transformer layers (ViT-Base)</p>
                                <ProgressBar mode="indeterminate" style="height: 6px"></ProgressBar>
                            </div>

                            <div v-else class="animate-fade-in">
                                <div class="inline-block px-3 py-1 bg-red-100 text-red-600 rounded-lg text-[10px] font-black uppercase tracking-widest mb-2">Result: Abnormal</div>
                                <h3 class="text-3xl font-black text-slate-900 mb-3">{{ e2eResults.diagnosis }}</h3>
                                <p class="text-slate-600 text-sm leading-relaxed mb-6">{{ e2eResults.details }}</p>
                                
                                <div class="flex items-center gap-4 p-4 bg-slate-50 rounded-2xl border border-slate-100">
                                    <div class="text-center px-4 border-r border-slate-200">
                                        <div class="text-[10px] font-bold text-slate-400 uppercase">Confidence</div>
                                        <div class="text-xl font-black text-blue-600">{{ e2eResults.confidence }}%</div>
                                    </div>
                                    <div class="flex-1">
                                        <div class="text-[10px] font-bold text-slate-400 uppercase mb-1">Risk Assessment</div>
                                        <div class="flex items-center gap-2">
                                            <div class="flex-1 h-2 bg-slate-200 rounded-full overflow-hidden">
                                                <div class="h-full bg-red-500 w-[70%]"></div>
                                            </div>
                                            <span class="text-xs font-bold text-red-500">High</span>
                                        </div>
                                    </div>
                                </div>
                                <Button label="New Scan" icon="pi pi-refresh" text class="mt-6 font-bold" @click="e2eImage=null; e2eStatus='idle'" />
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="glass p-6">
                    <h4 class="font-bold mb-2 flex items-center gap-2">
                        <i class="pi pi-info-circle text-blue-500"></i>
                        About E2E Model
                    </h4>
                    <p class="text-xs text-slate-500 leading-relaxed">Unlike the 2-step pipeline, the end-to-end model is trained on a massive dataset of annotated images directly, allowing it to pick up on subtle paper artifacts and grid distortions that might affect diagnosis.</p>
                </div>
                <div class="glass p-6">
                    <h4 class="font-bold mb-2 flex items-center gap-2">
                        <i class="pi pi-verified text-emerald-500"></i>
                        Attention Mapping
                    </h4>
                    <p class="text-xs text-slate-500 leading-relaxed">The model uses an integrated Gradient-weighted Class Activation Mapping (Grad-CAM) to highlight which part of the ECG strip most influenced the diagnosis.</p>
                </div>
            </div>
          </div>
        </div>

      </div>
    </main>
  </div>
</template>

<script>
// Required for PrimeIcons and Transitions
</script>

<style>
.animate-fade-in {
  animation: fadeIn 0.5s cubic-bezier(0.4, 0, 0.2, 1) forwards;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes scan {
  from { transform: translateY(0); }
  to { transform: translateY(380px); }
}

.animate-scan {
  animation: scan 1.5s ease-in-out infinite alternate;
}

/* Glassmorphism Refinements */
.glass {
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(226, 232, 240, 0.5);
  border-radius: 1.5rem;
  box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.04);
}

/* Custom Scrollbar */
::-webkit-scrollbar {
  width: 6px;
}
::-webkit-scrollbar-track {
  background: transparent;
}
::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
  background: #94a3b8;
}

/* PrimeVue Overrides */
.p-selectbutton .p-button {
    background: transparent;
    border: none;
    color: #64748b;
    font-size: 0.75rem;
    font-weight: 700;
}
.p-selectbutton .p-button.p-highlight {
    background: #ffffff !important;
    color: #0f172a !important;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06) !important;
}
.p-progressbar {
    background: #f1f5f9;
    border-radius: 10px;
}
.p-progressbar-value {
    background: #3b82f6;
}
.p-button {
    transition: all 0.2s;
}
.p-button:active {
    transform: scale(0.98);
}
</style>
