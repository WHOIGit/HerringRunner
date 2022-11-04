#!/usr/bin/env python
import argparse
import os
import random


parser = argparse.ArgumentParser()
parser.add_argument('SRC')
parser.add_argument('--train-ratio', default=0.9, type=float)
parser.add_argument('--val-ratio', default=0.1, type=float)
parser.add_argument('--test-ratio', default=0, type=float)
parser.add_argument('--neg-ratio',default=0.1, type=float)  # -1 is disable
parser.add_argument('--seed', default=random.randint(0,999999), type=int)
parser.add_argument('--outdir','-o', default='training-data/EXAMPLE_DIR')
parser.add_argument('--limit', nargs=2, metavar=('VIDEO','FRAMESLIMIT'), action='append', default=[])
parser.add_argument('--dotdotslash', action='store_true')

args = parser.parse_args()

print(f'Seed: {args.seed}')
random.seed(args.seed)

label_files = []
for (root, dirs, files) in os.walk(args.SRC):
    for f in files:
        if f.endswith('.txt') and '_' in f:
            videoid = f.rsplit('_',1)[0]
            label_file = os.path.join(root, f)
            bytes = os.path.getsize(label_file)
            count = round(bytes/34)
            image_file = label_file.replace('.txt','.jpg').replace('labels/','images/')#.replace('negatives/','')
            if not os.path.isfile(image_file):
                image_file = None
                
            label_files.append(dict(videoid=videoid, label_file=label_file, count=count, image_file=image_file))


videoids = sorted({d['videoid'] for d in label_files})
withfish = [d for d in label_files if d['count']!=0 and d['image_file']]
sansfish = [d for d in label_files if d['count']==0 and d['image_file']]
missing_jpgs = [d for d in label_files if d['image_file'] is None]

assert len(withfish)+len(sansfish) == len([d for d in label_files if d['image_file']])

def split(data, train_ratio, val_ratio, test_ratio=0, shuffle=False):
    assert 0 < train_ratio < 1
    if train_ratio + val_ratio < 1:
        test_ratio = 1-train_ratio-val_ratio
    assert train_ratio + val_ratio + test_ratio == 1
    idx1 = round(len(data)*train_ratio)
    idx2 = round(len(data)*(val_ratio+train_ratio))
    if shuffle:
        data = data.copy()
        random.shuffle(data)
    train = data[0:idx1]
    val = data[idx1:idx2]
    test = data[idx2:]
    return train,val,test
    

training_set = []
validation_set = []
test_set = []
missing_set = [d['label_file'] for d in missing_jpgs]
fishcount_total = 0

print(f'           video-id:  FILESET: <num-o-files> (withfish+sansfish)')
for videoid in videoids:
    # on a per-video basis evenly split up frames with and without fish into
    video_fishcount = sum([d['count'] for d in withfish if d['videoid']==videoid])
    withdata = [d['image_file'] for d in withfish if d['videoid']==videoid]
    sansdata = [d['image_file'] for d in sansfish if d['videoid']==videoid]
    missingjpg_count = sum([1 for d in missing_jpgs if d['videoid']==videoid])
    
    random.shuffle(withdata)
    random.shuffle(sansdata)
    
    for vid_lim,max_lim in args.limit:
        if videoid!=vid_lim: continue
        max_lim = int(max_lim)
        with_ratio = len(withdata)/(len(withdata)+len(sansdata))
        sans_ratio = len(sansdata)/(len(withdata)+len(sansdata))
        withdata = withdata[:round(max_lim*with_ratio)]
        sansdata = sansdata[:round(max_lim*sans_ratio)]
        video_fishcount = sum([d['count'] for d in withfish if d['videoid']==videoid and d['image_file'] in withdata])
    
    if args.neg_ratio >= 0:
        neg_ratio = round(len(withdata)*args.neg_ratio)
        sansdata = sansdata[:neg_ratio]
    
    withtups = split(withdata, args.train_ratio, args.val_ratio, args.test_ratio)
    sanstups = split(sansdata, args.train_ratio, args.val_ratio, args.test_ratio)
    
    training_set.extend(withtups[0]+sanstups[0])
    validation_set.extend(withtups[1]+sanstups[1])
    test_set.extend(withtups[2]+sanstups[2])
    fishcount_total += video_fishcount
    
    print(f'{videoid}: TRAINING:{len(withtups[0]+sanstups[0]):3} ({len(withtups[0]):3}+{len(sanstups[0]):3}), VALIDATION: {len(withtups[1]+sanstups[1])} ({len(withtups[1])}+{len(sanstups[1])}),{" TEST: {} ({}+{}),".format(len(withtups[2]+sanstups[2]),len(withtups[2]),len(sanstups[2])) if args.test_ratio else ""} FISHCOUNT: {video_fishcount}, MISSING_JPG_COUNT: {missingjpg_count} files')
print(f'             TOTALS: TRAINING: {len(training_set)}, VALIDATION: {len(validation_set)}, TEST: {len(test_set)}, ALL: {len(training_set)+len(validation_set)+len(test_set)}, FINAL_FISHCOUNT: {fishcount_total}, ALL_MISSING_JPGs: {len(missing_set)}')

training_set.sort()
validation_set.sort()
test_set.sort()
missing_set.sort()

if args.dotdotslash:
    training_set = [os.path.join('..',x) for x in training_set]
    validatoin_set = [os.path.join('..',x) for x in validation_set]
    test_set = [os.path.join('..',x) for x in test_set]

os.makedirs(args.outdir, exist_ok=True)

training_out = os.path.join(args.outdir,'training.txt')
training_out2 = os.path.join(args.outdir,'training.list')
print(f'Writing {training_out},{training_out2} (wc:{len(training_set)})')
with open(training_out, 'w') as o:
    o.writelines('\n'.join(training_set))
with open(training_out2, 'w') as o:
    frame_ids = [x.split('/')[-1].split('.')[0] for x in training_set]
    o.writelines('\n'.join(frame_ids))
    
validation_out = os.path.join(args.outdir,'validation.txt')
validation_out2 = os.path.join(args.outdir,'validation.list')
print(f'Writing {validation_out},{validation_out2} (wc:{len(validation_set)})')
with open(validation_out, 'w') as o:
    o.writelines('\n'.join(validation_set))
with open(validation_out2, 'w') as o:
    frame_ids = [x.split('/')[-1].split('.')[0] for x in validation_set]
    o.writelines('\n'.join(frame_ids))
    
test_out = os.path.join(args.outdir,'test.txt')
test_out2 = os.path.join(args.outdir,'test.list')
print(f'Writing {test_out},{test_out2} (wc:{len(test_set)})')    
with open(test_out, 'w') as o:
    o.writelines('\n'.join(test_set))
with open(test_out2, 'w') as o:
    frame_ids = [x.split('/')[-1].split('.')[0] for x in test_set]
    o.writelines('\n'.join(frame_ids))
    
missing_out = os.path.join(args.outdir,'missing_jpg.list')
print(f'Writing {missing_out} (wc:{len(missing_jpgs)})')    
with open(missing_out, 'w') as o:
    o.writelines('\n'.join(missing_set))


