import { OutputType, InputType } from '../types';
import { tasksInfo } from './tasksCatalog';

// Fallback modality mapping if tasksInfo not providing explicit input/output hints.
export interface TaskModalities {
  input: InputType[];
  output: OutputType[];
  multiInputSupport?: boolean; // true if the task can simultaneously consume >1 distinct modalities
}

export const taskModalities: Record<string, TaskModalities> = {};

for (const [id, info] of Object.entries(tasksInfo)) {
  const inp: InputType[] = [];
  let multiInputSupport = false;
  if (info.input) {
    const val = info.input.toLowerCase();
    const add = (m: InputType) => { if (!inp.includes(m)) inp.push(m); };
    if (val.includes('image+text')) { add('image'); add('text'); multiInputSupport = true; }
    else {
      if (val.includes('text')) add('text');
      if (val.includes('image')) add('image');
      if (val.includes('audio')) add('audio');
      if (val.includes('video')) add('video');
      if (val.includes('document')) add('document');
      if (val.includes('various')) { // treat as union; no guaranteed simultaneous multi support
        (['text','image','audio','video','document'] as InputType[]).forEach(add);
        multiInputSupport = false; // explicit: can't satisfy AND across all
      }
      // Enable multi-input only if we explicitly saw more than 1 and not 'various'
      if (!val.includes('various') && inp.length > 1) multiInputSupport = true;
    }
  }
  const out: OutputType[] = [];
  if (info.output) {
    const v = info.output.toLowerCase();
    const addOut = (o: OutputType) => { if (!out.includes(o)) out.push(o); };
    if (v.includes('text')) addOut('text');
    if (v.includes('image')) addOut('image');
    if (v.includes('audio')) addOut('audio');
    if (v.includes('video')) addOut('video');
    if (v.includes('3d')) addOut('3d');
    if (v.includes('embedding')) addOut('embedding');
    if (v.includes('mask')) addOut('mask');
    if (v.includes('boxes')) addOut('boxes');
    if (v.includes('scores')) addOut('scores');
    if (v.includes('points')) addOut('points');
    if (v.includes('segments')) addOut('segments');
    if (v.includes('depth')) addOut('depth');
    if (v.includes('image+text')) { addOut('image'); addOut('text'); }
  }
  taskModalities[id] = { input: inp.length? inp : ['text'], output: out.length? out : ['text'], multiInputSupport };
}
