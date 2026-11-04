import test from 'node:test';
import assert from 'node:assert/strict';
import { containRect, projectBox, detectionOpacity } from '../src/features/live/geometry.ts';
test('4:3 video is pillarboxed inside a 16:9 tile',()=>{assert.deepEqual(containRect(640,480,1280,720),{x:160,y:0,width:960,height:720});});
test('wide video is letterboxed inside a square tile',()=>{assert.deepEqual(containRect(1280,720,800,800),{x:0,y:175,width:800,height:450});});
test('normalized bbox includes the black-bar offset',()=>{assert.deepEqual(projectBox([.25,.25,.5,.5],containRect(640,480,1280,720)),{x:400,y:180,width:480,height:360});});
test('invalid media dimensions cannot produce NaN coordinates',()=>{assert.deepEqual(containRect(0,720,800,600),{x:0,y:0,width:0,height:0});});
test('detections fade after one second and disappear at three',()=>{assert.equal(detectionOpacity(500),1);assert.equal(detectionOpacity(1000),1);assert.equal(detectionOpacity(2000),.5);assert.equal(detectionOpacity(3000),0);assert.equal(detectionOpacity(5000),0);});
