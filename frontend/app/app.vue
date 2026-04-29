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
    plugins: {
      legend: { display: false },
      tooltip: { enabled: false }
    },
    scales: {
      x: { display: false },
      y: {
        display: true,
        min: -1.0,
        max: 2.0,
        grid: {
          color: 'rgba(255, 255, 255, 0.05)'
        },
        border: { display: false },
        ticks: { display: false }
      }
    }
  };
  
  updateChart();
  intervalId = setInterval(updateChart, 30);
});

onUnmounted(() => {
  clearInterval(intervalId);
});

const aiAnalysisStatus = ref('Analyzing');
const patientVitals = ref({
  hr: 72,
  bp: '120/80',
  o2: 98
});
</script>

<template>
  <div class="min-h-screen bg-slate-950 text-slate-100 flex overflow-hidden font-sans selection:bg-primary-500/30">
    <!-- Sidebar -->
    <aside class="w-20 lg:w-64 glass m-4 flex flex-col justify-between hidden md:flex">
      <div class="p-6">
        <div class="flex items-center gap-3 mb-10">
          <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center shadow-lg shadow-primary-500/30">
            <i class="pi pi-heart-fill text-white text-xl"></i>
          </div>
          <h1 class="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400 hidden lg:block tracking-tight">PhysioAI</h1>
        </div>
        
        <nav class="space-y-2">
          <a href="#" class="flex items-center gap-3 p-3 rounded-lg bg-primary-500/10 text-primary-400 border border-primary-500/20 transition-all">
            <i class="pi pi-home text-lg"></i>
            <span class="font-medium hidden lg:block">Dashboard</span>
          </a>
          <a href="#" class="flex items-center gap-3 p-3 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 transition-all">
            <i class="pi pi-users text-lg"></i>
            <span class="font-medium hidden lg:block">Patients</span>
          </a>
          <a href="#" class="flex items-center gap-3 p-3 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 transition-all">
            <i class="pi pi-chart-line text-lg"></i>
            <span class="font-medium hidden lg:block">Analytics</span>
          </a>
          <a href="#" class="flex items-center gap-3 p-3 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 transition-all">
            <i class="pi pi-cog text-lg"></i>
            <span class="font-medium hidden lg:block">Settings</span>
          </a>
        </nav>
      </div>
      
      <div class="p-6 border-t border-slate-700/50 flex items-center gap-3">
        <Avatar image="https://i.pravatar.cc/150?u=dr_smith" shape="circle" class="border-2 border-primary-500/50" />
        <div class="hidden lg:block">
          <p class="text-sm font-semibold">Dr. Sarah Jenkins</p>
          <p class="text-xs text-slate-400">Cardiologist</p>
        </div>
      </div>
    </aside>

    <!-- Main Content -->
    <main class="flex-1 p-4 overflow-y-auto">
      <header class="flex justify-between items-center mb-8 px-2">
        <div>
          <h2 class="text-2xl font-bold tracking-tight">Patient Monitor</h2>
          <p class="text-slate-400 text-sm">Live ECG streaming & AI inference</p>
        </div>
        <div class="flex items-center gap-4">
          <span class="relative flex h-3 w-3">
            <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
            <span class="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
          </span>
          <span class="text-sm text-slate-300 font-medium">Live Connection</span>
        </div>
      </header>

      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        <!-- Left Column: Patient Info & Vitals -->
        <div class="space-y-6 lg:col-span-1">
          <!-- Patient Profile -->
          <div class="glass p-6">
            <div class="flex justify-between items-start mb-6">
              <div class="flex gap-4">
                <Avatar image="https://i.pravatar.cc/150?u=patient_1" size="xlarge" shape="circle" class="shadow-lg" />
                <div>
                  <h3 class="text-xl font-bold">James Wilson</h3>
                  <p class="text-slate-400 text-sm">ID: #PT-84920</p>
                  <p class="text-slate-400 text-sm mt-1">Male, 58 yrs</p>
                </div>
              </div>
              <Button icon="pi pi-ellipsis-v" text rounded aria-label="Options" />
            </div>
            <div class="grid grid-cols-2 gap-4">
              <div class="bg-slate-900/50 p-3 rounded-lg border border-slate-700/50">
                <p class="text-xs text-slate-400 mb-1">Blood Type</p>
                <p class="font-semibold">O Positive</p>
              </div>
              <div class="bg-slate-900/50 p-3 rounded-lg border border-slate-700/50">
                <p class="text-xs text-slate-400 mb-1">Weight</p>
                <p class="font-semibold">82 kg</p>
              </div>
            </div>
          </div>

          <!-- Vitals Overview -->
          <div class="glass p-6 grid grid-cols-2 gap-4">
            <div class="col-span-2 sm:col-span-1 flex flex-col justify-center">
              <div class="flex items-center gap-2 mb-2">
                <i class="pi pi-heart text-red-500"></i>
                <span class="text-sm text-slate-400">Heart Rate</span>
              </div>
              <div class="flex items-end gap-2">
                <span class="text-4xl font-light tracking-tighter">{{ patientVitals.hr }}</span>
                <span class="text-sm text-slate-500 mb-1">bpm</span>
              </div>
            </div>
            <div class="col-span-2 sm:col-span-1 flex flex-col justify-center border-t sm:border-t-0 sm:border-l border-slate-700/50 sm:pl-4 pt-4 sm:pt-0">
              <div class="flex items-center gap-2 mb-2">
                <i class="pi pi-percentage text-blue-400"></i>
                <span class="text-sm text-slate-400">SpO2</span>
              </div>
              <div class="flex items-end gap-2">
                <span class="text-4xl font-light tracking-tighter">{{ patientVitals.o2 }}</span>
                <span class="text-sm text-slate-500 mb-1">%</span>
              </div>
            </div>
            <div class="col-span-2 pt-4 border-t border-slate-700/50">
              <div class="flex items-center gap-2 mb-2">
                <i class="pi pi-chart-bar text-purple-400"></i>
                <span class="text-sm text-slate-400">Blood Pressure</span>
              </div>
              <div class="flex items-end gap-2">
                <span class="text-3xl font-light tracking-tighter">{{ patientVitals.bp }}</span>
                <span class="text-sm text-slate-500 mb-1">mmHg</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Right Column: ECG & AI Analysis -->
        <div class="space-y-6 lg:col-span-2">
          
          <!-- Main ECG Viewer -->
          <div class="glass p-6 relative overflow-hidden">
            <!-- Background grid effect -->
            <div class="absolute inset-0 opacity-[0.03] pointer-events-none" style="background-image: linear-gradient(#fff 1px, transparent 1px), linear-gradient(90deg, #fff 1px, transparent 1px); background-size: 20px 20px;"></div>
            
            <div class="flex justify-between items-center mb-6 relative z-10">
              <div class="flex items-center gap-3">
                <h3 class="text-xl font-bold">Lead II ECG</h3>
                <Tag value="25 mm/s" severity="secondary" class="text-xs" />
              </div>
              <div class="flex gap-2">
                <Button icon="pi pi-search-plus" size="small" text rounded aria-label="Zoom In" class="text-slate-400 hover:text-white" />
                <Button icon="pi pi-pause" size="small" text rounded aria-label="Pause" class="text-slate-400 hover:text-white" />
                <Button icon="pi pi-cog" size="small" text rounded aria-label="Settings" class="text-slate-400 hover:text-white" />
              </div>
            </div>
            
            <div class="h-64 w-full relative z-10">
              <Chart type="line" :data="chartData" :options="chartOptions" class="h-full w-full" />
            </div>
          </div>

          <!-- AI Model Analysis -->
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="glass p-6 bg-gradient-to-br from-slate-800/80 to-slate-900/80">
              <div class="flex items-center gap-3 mb-4">
                <i class="pi pi-microchip-ai text-primary-400 text-xl"></i>
                <h3 class="text-lg font-semibold">PhysioWave Transformer</h3>
              </div>
              <p class="text-sm text-slate-400 mb-6">Real-time contextual rhythm analysis using Multi-Scale Wavelet Attention.</p>
              
              <div class="space-y-4">
                <div>
                  <div class="flex justify-between text-sm mb-1">
                    <span>Normal Sinus Rhythm (N)</span>
                    <span class="font-mono text-primary-400">98.5%</span>
                  </div>
                  <ProgressBar :value="98.5" :showValue="false" style="height: 6px" class="bg-slate-700"></ProgressBar>
                </div>
                <div>
                  <div class="flex justify-between text-sm mb-1">
                    <span class="text-slate-400">Supraventricular (S)</span>
                    <span class="font-mono text-slate-400">0.8%</span>
                  </div>
                  <ProgressBar :value="0.8" :showValue="false" style="height: 6px" class="bg-slate-800"></ProgressBar>
                </div>
                <div>
                  <div class="flex justify-between text-sm mb-1">
                    <span class="text-slate-400">Ventricular Ectopic (V)</span>
                    <span class="font-mono text-slate-400">0.5%</span>
                  </div>
                  <ProgressBar :value="0.5" :showValue="false" style="height: 6px" class="bg-slate-800"></ProgressBar>
                </div>
              </div>
            </div>

            <!-- Alerts / Log -->
            <div class="glass p-6">
              <h3 class="text-lg font-semibold mb-4">Clinical Insights</h3>
              <div class="space-y-4">
                <div class="p-3 rounded-lg bg-green-500/10 border border-green-500/20 flex gap-3">
                  <i class="pi pi-check-circle text-green-400 mt-0.5"></i>
                  <div>
                    <p class="text-sm font-medium text-green-100">Stable Rhythm Detected</p>
                    <p class="text-xs text-green-400/70 mt-1">No significant morphological anomalies found in the last 2 hours.</p>
                  </div>
                </div>
                <div class="p-3 rounded-lg bg-slate-800/50 border border-slate-700/50 flex gap-3">
                  <i class="pi pi-info-circle text-blue-400 mt-0.5"></i>
                  <div>
                    <p class="text-sm font-medium">Baseline Update</p>
                    <p class="text-xs text-slate-400 mt-1">Model successfully recalibrated to patient's diurnal baseline.</p>
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
/* Global specific overrides or PrimeVue PT classes could go here if not using Tailwind fully */
</style>
