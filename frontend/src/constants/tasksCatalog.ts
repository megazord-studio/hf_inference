// Curated task catalog grouped for UI filtering (derived from backend TASKS set)
export interface TaskCategory {
  id: string;
  label: string;
  description: string;
  tasks: string[]; // raw task ids
}

export interface TaskInfo {
  id: string;            // raw HF task id
  label: string;         // human friendly label
  description: string;   // short help text
  input?: string;        // primary input modality summary
  output?: string;       // primary output modality summary
  aliases?: string[];    // alternative names visible in search
  category?: string;     // category id reference
  highlight?: boolean;   // if true visually emphasize (e.g. multimodal)
}

export const taskCategories: TaskCategory[] = [
  {
    id: 'language',
    label: 'Language',
    description: 'Text understanding & generation',
    tasks: [
      'text-generation','summarization','translation','question-answering','table-question-answering',
      'text-classification','token-classification','zero-shot-classification','fill-mask','text-ranking',
      'sentence-similarity','feature-extraction'
    ]
  },
  {
    id: 'vision-understanding',
    label: 'Vision Understanding',
    description: 'Image analysis & detection',
    tasks: [
      'image-classification','object-detection','image-segmentation','zero-shot-image-classification','zero-shot-object-detection',
      'keypoint-detection','depth-estimation'
    ]
  },
  {
    id: 'multimodal-reasoning',
    label: 'Vision + Language',
    description: 'Cross-modal reasoning & captioning',
    tasks: ['image-to-text','image-text-to-text']
  },
  {
    id: 'vision-generation',
    label: 'Vision Generation',
    description: 'Image / 3D generation & enhancement',
    tasks: [
      'text-to-image','image-to-image','image-super-resolution','image-restoration','image-to-3d','text-to-3d'
    ]
  },
  {
    id: 'video',
    label: 'Video',
    description: 'Video generation & conversion',
    tasks: ['text-to-video','image-to-video']
  },
  {
    id: 'audio',
    label: 'Audio & Speech',
    description: 'Speech recognition, generation & audio processing',
    tasks: [
      'automatic-speech-recognition','text-to-speech','audio-classification','audio-to-audio','text-to-audio','audio-text-to-text','voice-activity-detection'
    ]
  },
  {
    id: 'forecasting',
    label: 'Forecasting',
    description: 'Time series prediction',
    tasks: ['time-series-forecasting']
  },
  {
    id: 'retrieval',
    label: 'Retrieval',
    description: 'Cross-modal & document retrieval',
    tasks: ['visual-document-retrieval']
  },
  {
    id: 'generalist',
    label: 'Generalist',
    description: 'Flexible any-to-any models',
    tasks: ['any-to-any']
  }
];

// Metadata for tasks (only those in curated catalog)
const buildTaskInfo = (): TaskInfo[] => {
  const info: TaskInfo[] = [
    { id: 'text-generation', label: 'Text Generation', description: 'Generate or continue text', input: 'text', output: 'text', aliases: ['llm','generate text'], category: 'language' },
    { id: 'summarization', label: 'Summarization', description: 'Condense long text', input: 'text', output: 'text', category: 'language' },
    { id: 'translation', label: 'Translation', description: 'Translate between languages', input: 'text', output: 'text', category: 'language' },
    { id: 'question-answering', label: 'QA (Extractive)', description: 'Answer questions from context', input: 'text', output: 'text', aliases: ['qa'], category: 'language' },
    { id: 'table-question-answering', label: 'Table QA', description: 'Answer questions from tables', input: 'text', output: 'text', aliases: ['table qa'], category: 'language' },
    { id: 'text-classification', label: 'Text Classification', description: 'Assign labels to text', input: 'text', output: 'labels', category: 'language' },
    { id: 'token-classification', label: 'Token Classification', description: 'Per-token labeling (NER)', input: 'text', output: 'tags', aliases: ['ner'], category: 'language' },
    { id: 'zero-shot-classification', label: 'Zero-shot Text Classification', description: 'Classify text without training labels', input: 'text', output: 'labels', category: 'language' },
    { id: 'fill-mask', label: 'Fill Mask', description: 'Predict masked tokens', input: 'text', output: 'text', category: 'language' },
    { id: 'text-ranking', label: 'Text Ranking', description: 'Rank passages or documents', input: 'text', output: 'scores', category: 'language' },
    { id: 'sentence-similarity', label: 'Sentence Similarity', description: 'Embedding similarity scoring', input: 'text', output: 'embedding', aliases: ['embeddings'], category: 'language' },
    { id: 'feature-extraction', label: 'Text Embeddings', description: 'General embeddings for search', input: 'text', output: 'embedding', aliases: ['embeddings'], category: 'language' },

    { id: 'image-classification', label: 'Image Classification', description: 'Identify classes in an image', input: 'image', output: 'labels', category: 'vision-understanding' },
    { id: 'object-detection', label: 'Object Detection', description: 'Locate objects (bounding boxes)', input: 'image', output: 'boxes', category: 'vision-understanding' },
    { id: 'image-segmentation', label: 'Image Segmentation', description: 'Per-pixel class masks', input: 'image', output: 'mask', category: 'vision-understanding' },
    { id: 'zero-shot-image-classification', label: 'Zero-shot Image Classification', description: 'Classify images without finetune', input: 'image', output: 'labels', category: 'vision-understanding' },
    { id: 'zero-shot-object-detection', label: 'Zero-shot Object Detection', description: 'Detect objects without finetune', input: 'image', output: 'boxes', category: 'vision-understanding' },
    { id: 'keypoint-detection', label: 'Keypoint Detection', description: 'Detect keypoints/landmarks', input: 'image', output: 'points', category: 'vision-understanding' },
    { id: 'depth-estimation', label: 'Depth Estimation', description: 'Predict depth from image', input: 'image', output: 'depth map', category: 'vision-understanding' },

    { id: 'image-to-text', label: 'Image Captioning', description: 'Generate description of an image', input: 'image', output: 'text', category: 'multimodal-reasoning', highlight: true },
    { id: 'image-text-to-text', label: 'Vision-Language Reasoning', description: 'Answer questions / reasoning over image + text', input: 'image+text', output: 'text', aliases: ['vqa','vision language'], category: 'multimodal-reasoning', highlight: true },

    { id: 'text-to-image', label: 'Text → Image', description: 'Generate images from text prompts', input: 'text', output: 'image', category: 'vision-generation' },
    { id: 'image-to-image', label: 'Image → Image', description: 'Transform or enhance images', input: 'image', output: 'image', category: 'vision-generation' },
    { id: 'image-super-resolution', label: 'Super Resolution', description: 'Upscale images', input: 'image', output: 'image', category: 'vision-generation' },
    { id: 'image-restoration', label: 'Image Restoration', description: 'Denoise / repair images', input: 'image', output: 'image', category: 'vision-generation' },
    { id: 'image-to-3d', label: 'Image → 3D', description: 'Generate 3D from an image', input: 'image', output: '3D', category: 'vision-generation' },
    { id: 'text-to-3d', label: 'Text → 3D', description: 'Generate 3D from text', input: 'text', output: '3D', category: 'vision-generation' },

    { id: 'text-to-video', label: 'Text → Video', description: 'Generate video from text', input: 'text', output: 'video', category: 'video' },
    { id: 'image-to-video', label: 'Image → Video', description: 'Generate video from image', input: 'image', output: 'video', category: 'video' },

    { id: 'automatic-speech-recognition', label: 'Speech Recognition', description: 'Transcribe spoken audio', input: 'audio', output: 'text', aliases: ['asr'], category: 'audio' },
    { id: 'text-to-speech', label: 'Text → Speech', description: 'Generate speech from text', input: 'text', output: 'audio', aliases: ['tts'], category: 'audio' },
    { id: 'audio-classification', label: 'Audio Classification', description: 'Label audio events', input: 'audio', output: 'labels', category: 'audio' },
    { id: 'audio-to-audio', label: 'Audio Enhancement', description: 'Transform audio (denoise etc.)', input: 'audio', output: 'audio', category: 'audio' },
    { id: 'text-to-audio', label: 'Text → Audio (Music/SFX)', description: 'Generate music or sound from text', input: 'text', output: 'audio', category: 'audio' },
    { id: 'audio-text-to-text', label: 'Audio → Text (Advanced)', description: 'Complex audio transcription / reasoning', input: 'audio', output: 'text', category: 'audio' },
    { id: 'voice-activity-detection', label: 'Voice Activity Detection', description: 'Detect speech segments', input: 'audio', output: 'segments', aliases: ['vad'], category: 'audio' },

    { id: 'time-series-forecasting', label: 'Time Series Forecasting', description: 'Predict future values', input: 'series', output: 'series', category: 'forecasting' },
    { id: 'visual-document-retrieval', label: 'Visual Document Retrieval', description: 'Retrieve documents by visual content', input: 'image+text', output: 'docs', category: 'retrieval' },
    { id: 'any-to-any', label: 'Any ↔ Any', description: 'Generalist flexible modality model', input: 'various', output: 'various', category: 'generalist', highlight: true },
  ];
  return info;
};

export const tasksInfo: Record<string, TaskInfo> = Object.fromEntries(buildTaskInfo().map(t => [t.id, t]));

export const taskToCategory: Record<string,string> = Object.fromEntries(
  taskCategories.flatMap(cat => cat.tasks.map(t => [t, cat.id]))
);
