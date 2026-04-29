<script setup>
import { ref, onMounted, onUnmounted } from 'vue';
import Chart from 'primevue/chart';
import Button from 'primevue/button';
import Avatar from 'primevue/avatar';
import ProgressBar from 'primevue/progressbar';
import Tag from 'primevue/tag';
import FileUpload from 'primevue/fileupload';
import SelectButton from 'primevue/selectbutton';

const activeTab = ref('dashboard');

// --- Dashboard Live Data ---
const chartData = ref(null);
const chartOptions = ref(null);

let timeIndex = 0;
const generateECGDataPoint = () => {
  const t = timeIndex % 100;
  timeIndex++;
  // Basic NSR mock
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
      borderColor: '#3b82f6', // primary-500
      tension: 0.4,
      borderWidth: 2.5,
      pointRadius: 0,
    }]
  };
};

// --- Comparison & Explorer Data ---
const viewOptions = ref(['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']);
const selectedView = ref('Normal (N)');

const comparisonData = ref(null);
const comparisonOptions = ref(null);

const generateStaticWave = (type) => {
    const arr = [];
    for(let t=0; t<100; t++) {
        if(type === 'Normal (N)') {
            if (t > 10 && t < 20) arr.push(0.2 * Math.sin((t - 10) * Math.PI / 10));
            else if (t === 30) arr.push(-0.3);
            else if (t === 32) arr.push(1.5);
            else if (t === 34) arr.push(-0.4);
            else if (t > 50 && t < 70) arr.push(0.4 * Math.sin((t - 50) * Math.PI / 20));
            else arr.push(0);
        } else if (type === 'Ventricular (V)') {
             // PVC - wide and bizarre QRS
             if (t > 25 && t < 45) arr.push(2.0 * Math.sin((t - 25) * Math.PI / 20));
             else if (t > 50 && t < 80) arr.push(-0.8 * Math.sin((t - 50) * Math.PI / 30));
             else arr.push(0);
        } else {
             // Fallback/other shapes
             if (t === 30) arr.push(-0.1);
             else if (t === 32) arr.push(0.8);
             else if (t === 34) arr.push(-0.5);
             else arr.push(0);
        }
    }
    return arr;
};

const loadComparison = () => {
   comparisonData.value = {
    labels: Array.from({ length: 100 }, (_, i) => i),
    datasets: [
      {
        label: 'Normal Baseline',
        data: generateStaticWave('Normal (N)'),
        borderColor: '#94a3b8', 
        borderDash: [5, 5],
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 0,
      },
      {
        label: selectedView.value,
        data: generateStaticWave(selectedView.value),
        borderColor: '#ef4444', // red-500 for anomaly
        tension: 0.4,
        borderWidth: 3,
        pointRadius: 0,
      }
    ]
  };
}

// --- File Upload State ---
const uploadedImage = ref(null);
const isAnalyzing = ref(false);
const analysisComplete = ref(false);

const onUpload = (event) => {
    const file = event.files[0];
    const reader = new FileReader();
    reader.onload = (e) => {
        uploadedImage.value = e.target.result;
        isAnalyzing.value = true;
        analysisComplete.value = false;
        // Simulate AI analysis delay
        setTimeout(() => {
            isAnalyzing.value = false;
            analysisComplete.value = true;
        }, 3000);
    };
    reader.readAsDataURL(file);
};

onMounted(() => {
  chartOptions.value = {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false }, tooltip: { enabled: false } },
    scales: {
      x: { display: false },
      y: { display: true, min: -1.0, max: 2.0, grid: { color: 'rgba(0, 0, 0, 0.05)' }, border: { display: false }, ticks: { display: false } }
    }
  };
  
  comparisonOptions.value = {
    animation: true,
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: true, position: 'top' } },
    scales: {
      x: { display: false },
      y: { display: true, min: -1.5, max: 2.5, grid: { color: 'rgba(0, 0, 0, 0.05)' } }
    }
  };

  updateChart();
  intervalId = setInterval(updateChart, 30);
  loadComparison();
});

onUnmounted(() => {
  clearInterval(intervalId);
});

const patientVitals = ref({ hr: 72, bp: '120/80', o2: 98 });
</script>

<template>
  <div class="min-h-screen bg-slate-50 text-slate-800 flex overflow-hidden font-sans">
    <!-- Sidebar -->
    <aside class="w-20 lg:w-64 bg-white border-r border-slate-200 flex flex-col justify-between hidden md:flex z-10 shadow-[4px_0_24px_rgba(0,0,0,0.02)]">
      <div class="p-6">
        <div class="flex items-center gap-3 mb-10">
          <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center shadow-lg shadow-primary-500/30">
            <i class="pi pi-heart-fill text-white text-xl"></i>
          </div>
          <h1 class="text-xl font-bold text-slate-900 hidden lg:block tracking-tight">PhysioAI</h1>
        </div>
        
        <nav class="space-y-2">
          <button @click="activeTab = 'dashboard'" :class="['w-full flex items-center gap-3 p-3 rounded-lg transition-all text-left', activeTab === 'dashboard' ? 'bg-primary-50 text-primary-600 font-semibold' : 'text-slate-500 hover:bg-slate-100']">
            <i class="pi pi-home text-lg"></i>
            <span class="hidden lg:block">Dashboard</span>
          </button>
          <button @click="activeTab = 'explorer'" :class="['w-full flex items-center gap-3 p-3 rounded-lg transition-all text-left', activeTab === 'explorer' ? 'bg-primary-50 text-primary-600 font-semibold' : 'text-slate-500 hover:bg-slate-100']">
            <i class="pi pi-book text-lg"></i>
            <span class="hidden lg:block">Class Explorer</span>
          </button>
          <button @click="activeTab = 'upload'" :class="['w-full flex items-center gap-3 p-3 rounded-lg transition-all text-left', activeTab === 'upload' ? 'bg-primary-50 text-primary-600 font-semibold' : 'text-slate-500 hover:bg-slate-100']">
            <i class="pi pi-cloud-upload text-lg"></i>
            <span class="hidden lg:block">Upload & Predict</span>
          </button>
        </nav>
      </div>
      
      <div class="p-6 border-t border-slate-100 flex items-center gap-3">
        <Avatar image="https://i.pravatar.cc/150?u=dr_smith" shape="circle" class="border-2 border-white shadow-md" />
        <div class="hidden lg:block">
          <p class="text-sm font-bold text-slate-900">Dr. Sarah Jenkins</p>
          <p class="text-xs text-slate-500">Cardiologist</p>
        </div>
      </div>
    </aside>

    <!-- Main Content -->
    <main class="flex-1 overflow-y-auto bg-slate-50 relative">
      <!-- Top header decorative background -->
      <div class="absolute top-0 left-0 right-0 h-64 bg-gradient-to-b from-primary-50/50 to-transparent pointer-events-none"></div>

      <div class="p-8 relative z-10 max-w-7xl mx-auto">
        
        <!-- Tab: Dashboard -->
        <div v-if="activeTab === 'dashboard'" class="animate-fade-in">
          <header class="flex justify-between items-center mb-8">
            <div>
              <h2 class="text-3xl font-bold tracking-tight text-slate-900">Patient Monitor</h2>
              <p class="text-slate-500">Live ECG streaming & AI inference</p>
            </div>
            <div class="flex items-center gap-3 bg-white px-4 py-2 rounded-full shadow-sm border border-slate-100">
              <span class="relative flex h-2.5 w-2.5">
                <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500"></span>
              </span>
              <span class="text-sm font-semibold text-slate-700">Live Connection</span>
            </div>
          </header>

          <div class="grid grid-cols-1 xl:grid-cols-3 gap-6">
            <!-- Left Col: Vitals -->
            <div class="space-y-6">
              <div class="glass p-6">
                <div class="flex gap-4 mb-6">
                  <Avatar image="https://i.pravatar.cc/150?u=patient_1" size="xlarge" shape="circle" class="shadow-md" />
                  <div>
                    <h3 class="text-xl font-bold text-slate-900">James Wilson</h3>
                    <p class="text-slate-500 text-sm">ID: #PT-84920 &bull; Male, 58 yrs</p>
                  </div>
                </div>
                <div class="grid grid-cols-2 gap-4">
                  <div class="bg-slate-50 p-4 rounded-xl border border-slate-100">
                    <div class="flex items-center gap-2 mb-1">
                      <i class="pi pi-heart text-red-500"></i>
                      <span class="text-xs font-semibold text-slate-500 uppercase">Heart Rate</span>
                    </div>
                    <div class="flex items-end gap-1">
                      <span class="text-3xl font-bold text-slate-800">{{ patientVitals.hr }}</span>
                      <span class="text-xs text-slate-500 mb-1">bpm</span>
                    </div>
                  </div>
                  <div class="bg-slate-50 p-4 rounded-xl border border-slate-100">
                    <div class="flex items-center gap-2 mb-1">
                      <i class="pi pi-percentage text-blue-500"></i>
                      <span class="text-xs font-semibold text-slate-500 uppercase">SpO2</span>
                    </div>
                    <div class="flex items-end gap-1">
                      <span class="text-3xl font-bold text-slate-800">{{ patientVitals.o2 }}</span>
                      <span class="text-xs text-slate-500 mb-1">%</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <!-- Right Col: ECG -->
            <div class="xl:col-span-2 space-y-6">
              <div class="glass p-6 relative overflow-hidden">
                <!-- ECG Grid Background -->
                <div class="absolute inset-0 opacity-[0.15]" style="background-image: linear-gradient(#f43f5e 1px, transparent 1px), linear-gradient(90deg, #f43f5e 1px, transparent 1px); background-size: 20px 20px;"></div>
                <div class="flex justify-between items-center mb-4 relative z-10">
                  <h3 class="font-bold text-lg text-slate-900">Lead II ECG</h3>
                  <Tag value="25 mm/s" severity="secondary" class="text-xs" />
                </div>
                <div class="h-64 w-full relative z-10 bg-white/50 rounded-lg backdrop-blur-sm border border-slate-200/50">
                  <Chart type="line" :data="chartData" :options="chartOptions" class="h-full w-full" />
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Tab: Class Explorer -->
        <div v-if="activeTab === 'explorer'" class="animate-fade-in">
          <header class="mb-8">
            <h2 class="text-3xl font-bold tracking-tight text-slate-900">Class Explorer & Comparison</h2>
            <p class="text-slate-500">Compare different arrhythmia classes against a normal baseline.</p>
          </header>

          <div class="glass p-6">
            <div class="mb-6">
              <p class="font-medium text-slate-700 mb-3">Select Class to Compare:</p>
              <SelectButton v-model="selectedView" :options="viewOptions" @change="loadComparison" class="flex flex-wrap" />
            </div>
            
            <div class="h-80 w-full bg-white rounded-xl border border-slate-100 p-4 shadow-inner">
              <Chart type="line" :data="comparisonData" :options="comparisonOptions" class="h-full w-full" />
            </div>

            <div class="mt-6 bg-blue-50 border border-blue-100 p-4 rounded-xl flex gap-4">
              <i class="pi pi-info-circle text-blue-500 text-xl mt-1"></i>
              <div>
                <h4 class="font-bold text-blue-900">Understanding {{ selectedView }}</h4>
                <p class="text-sm text-blue-800 mt-1">
                  <span v-if="selectedView === 'Normal (N)'">The normal sinus rhythm originates from the SA node. The QRS complex is narrow, and P waves are consistent.</span>
                  <span v-if="selectedView === 'Ventricular (V)'">A Premature Ventricular Contraction (PVC) shows a wide, bizarre QRS complex with no preceding P wave, originating from the ventricles.</span>
                  <span v-if="selectedView.includes('S')">Supraventricular premature beats originate above the ventricles. The QRS is usually narrow, but the P wave may be premature or abnormally shaped.</span>
                  <span v-if="selectedView.includes('F')">Fusion beats occur when a supraventricular impulse and a ventricular impulse coincide, producing a hybrid QRS complex.</span>
                  <span v-if="selectedView.includes('Q')">Unclassifiable or paced beats that do not fit the standard morphological patterns.</span>
                </p>
              </div>
            </div>
          </div>
        </div>

        <!-- Tab: Upload & Predict -->
        <div v-if="activeTab === 'upload'" class="animate-fade-in">
          <header class="mb-8">
            <h2 class="text-3xl font-bold tracking-tight text-slate-900">Diagnostic Tool</h2>
            <p class="text-slate-500">Upload an ECG image or signal file (CSV/JSON) for instant AI classification.</p>
          </header>

          <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <!-- Upload Area -->
            <div class="glass p-6 flex flex-col items-center justify-center min-h-[400px]">
              <div v-if="!uploadedImage" class="w-full">
                <FileUpload mode="advanced" accept="image/*" :maxFileSize="5000000" @select="onUpload" customUpload :auto="true" chooseLabel="Select ECG Image">
                  <template #empty>
                    <div class="flex flex-col items-center justify-center py-10">
                      <i class="pi pi-cloud-upload text-5xl text-slate-300 mb-4"></i>
                      <p class="text-slate-500">Drag and drop an ECG image here to analyze.</p>
                      <p class="text-xs text-slate-400 mt-2">Supports PNG, JPG. Signal extraction will be applied automatically.</p>
                    </div>
                  </template>
                </FileUpload>
              </div>

              <div v-else class="w-full flex flex-col items-center animate-fade-in">
                <img :src="uploadedImage" alt="Uploaded ECG" class="max-h-64 rounded-xl border border-slate-200 shadow-sm mb-6 object-contain bg-white p-2" />
                <Button label="Upload Another" icon="pi pi-refresh" text @click="uploadedImage = null; analysisComplete = false" />
              </div>
            </div>

            <!-- Results Area -->
            <div class="glass p-6">
              <h3 class="text-xl font-bold text-slate-900 mb-6">Analysis Results</h3>
              
              <div v-if="!uploadedImage" class="flex flex-col items-center justify-center h-64 text-slate-400">
                <i class="pi pi-microchip-ai text-4xl mb-3 opacity-50"></i>
                <p>Waiting for input data...</p>
              </div>

              <div v-else-if="isAnalyzing" class="flex flex-col items-center justify-center h-64 space-y-4">
                <i class="pi pi-spin pi-spinner text-4xl text-primary-500"></i>
                <p class="text-slate-600 font-medium animate-pulse">Extracting 1D signal from image & running inference...</p>
              </div>

              <div v-else-if="analysisComplete" class="animate-fade-in">
                <div class="bg-red-50 border border-red-200 p-4 rounded-xl flex gap-4 mb-6">
                  <div class="w-12 h-12 bg-red-100 rounded-full flex items-center justify-center shrink-0">
                    <i class="pi pi-exclamation-triangle text-red-500 text-xl"></i>
                  </div>
                  <div>
                    <h4 class="text-lg font-bold text-red-900">Premature Ventricular Contraction (V)</h4>
                    <p class="text-sm text-red-700 mt-1">The model detected an abnormal morphology consistent with a ventricular ectopic beat with 94.2% confidence.</p>
                  </div>
                </div>

                <h4 class="font-semibold text-slate-700 mb-3">Confidence Distribution</h4>
                <div class="space-y-4">
                  <div>
                    <div class="flex justify-between text-sm mb-1">
                      <span class="font-medium text-slate-800">Ventricular (V)</span>
                      <span class="font-bold text-red-500">94.2%</span>
                    </div>
                    <ProgressBar :value="94.2" :showValue="false" style="height: 8px"></ProgressBar>
                  </div>
                  <div>
                    <div class="flex justify-between text-sm mb-1">
                      <span class="text-slate-500">Normal (N)</span>
                      <span class="text-slate-500">4.1%</span>
                    </div>
                    <ProgressBar :value="4.1" :showValue="false" style="height: 8px" class="opacity-50"></ProgressBar>
                  </div>
                  <div>
                    <div class="flex justify-between text-sm mb-1">
                      <span class="text-slate-500">Supraventricular (S)</span>
                      <span class="text-slate-500">1.2%</span>
                    </div>
                    <ProgressBar :value="1.2" :showValue="false" style="height: 8px" class="opacity-50"></ProgressBar>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

      </div>
    </main>
  </div>
</template>

<style>
.animate-fade-in {
  animation: fadeIn 0.4s ease-out forwards;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(5px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
