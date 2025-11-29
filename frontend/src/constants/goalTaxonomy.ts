import type { GoalCategory } from '../types';

// Initial goal-oriented taxonomy mapping user intents (natural language goals) to HF tasks.
// This can evolve; keep IDs stable for persistence.
export const goalCategories: GoalCategory[] = [
  {
    id: 'text-understanding',
    label: 'Understand / Transform Text',
    subcategories: [
      {
        id: 'summarize',
        label: 'Summarize text',
        description: 'Condense long passages into concise summaries.',
        tasks: ['summarization'],
        input_types: ['text'],
        output_types: ['text'],
        examples: ['Summarize this article about climate change']
      },
      {
        id: 'translate',
        label: 'Translate text',
        description: 'Convert text from one language to another.',
        tasks: ['translation','text2text-generation'],
        input_types: ['text'],
        output_types: ['text'],
        examples: ['Translate this German text to English']
      },
      {
        id: 'qa',
        label: 'Answer questions',
        description: 'Extract or generate answers given context.',
        tasks: ['question-answering','table-question-answering'],
        input_types: ['text'],
        output_types: ['text'],
        examples: ['Answer: What year was the company founded?']
      },
      {
        id: 'classify-text',
        label: 'Classify / Tag',
        description: 'Assign labels or sentiments to text.',
        tasks: ['zero-shot-classification','sentiment-analysis','token-classification'],
        input_types: ['text'],
        output_types: ['text'],
        examples: ['Is this review positive or negative?']
      },
      {
        id: 'embed-text',
        label: 'Vector embeddings',
        description: 'Create embeddings for semantic search or similarity.',
        tasks: ['feature-extraction'],
        input_types: ['text'],
        output_types: ['embedding'],
        examples: ['Embed this sentence for similarity search']
      }
    ]
  },
  {
    id: 'image-understanding',
    label: 'Understand / Transform Images',
    subcategories: [
      {
        id: 'describe-image',
        label: 'Describe an image',
        description: 'Generate a caption or textual description of an image.',
        tasks: ['image-to-text','image-text-to-text'],
        input_types: ['image','multimodal'],
        output_types: ['text'],
        examples: ['Describe what is in this photo']
      },
      {
        id: 'detect-objects',
        label: 'Detect objects',
        description: 'Identify and localize objects within the image.',
        tasks: ['object-detection'],
        input_types: ['image'],
        output_types: ['boxes'],
        examples: ['Find all people and cars in this image']
      },
      {
        id: 'segment-image',
        label: 'Segment / Mask',
        description: 'Produce segmentation masks for entities.',
        tasks: ['image-segmentation','mask-generation'],
        input_types: ['image'],
        output_types: ['mask'],
        examples: ['Segment the dog in this picture']
      },
      {
        id: 'depth-estimate',
        label: 'Depth estimation',
        description: 'Infer per-pixel depth map from image.',
        tasks: ['depth-estimation'],
        input_types: ['image'],
        output_types: ['depth'],
        examples: ['Generate depth map of this scene']
      },
      {
        id: 'image-to-image',
        label: 'Transform image',
        description: 'Modify or enhance an image based on instructions.',
        tasks: ['image-to-image'],
        input_types: ['image'],
        output_types: ['image'],
        examples: ['Improve sharpness of this image']
      }
    ]
  },
  {
    id: 'image-generation',
    label: 'Generate Images from Text',
    subcategories: [
      {
        id: 'text-to-image',
        label: 'Text to image',
        description: 'Create new images from textual prompts.',
        tasks: ['text-to-image'],
        input_types: ['text'],
        output_types: ['image'],
        examples: ['A watercolor painting of a mountain at sunrise']
      }
    ]
  },
  {
    id: 'multimodal-reasoning',
    label: 'Multimodal Reasoning',
    subcategories: [
      {
        id: 'vision-language',
        label: 'Image + Text reasoning',
        description: 'Answer questions or analyze with both image and text context.',
        tasks: ['image-text-to-text'],
        input_types: ['multimodal','image','text'],
        output_types: ['text'],
        examples: ['Given this image and prompt, explain the safety issues']
      }
    ]
  }
];

export const goalCategoryIndex: Record<string, GoalCategory> = Object.fromEntries(goalCategories.map(c => [c.id, c]));

export function getTasksForGoal(categoryId?: string | null, subcategoryId?: string | null): string[] {
  if (!categoryId) return [];
  const cat = goalCategoryIndex[categoryId];
  if (!cat) return [];
  if (!subcategoryId) return Array.from(new Set(cat.subcategories.flatMap(s => s.tasks)));
  const sub = cat.subcategories.find(s => s.id === subcategoryId);
  return sub ? sub.tasks : [];
}

export function getSubcategories(categoryId?: string | null) {
  if (!categoryId) return [];
  return goalCategoryIndex[categoryId]?.subcategories || [];
}

