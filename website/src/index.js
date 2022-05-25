import _ from 'lodash';
import img_splash from './assets/images/splash-thumbnail.png';
import img_summaryfig from './assets/images/fig_design_summaryfig-cogsci2022.jpg';
// import vid_training from './stimuli/dynamics/258_c_ed_vbr2.mp4';
import d2761 from './stimuli/dynamics/276_1t.mp4';
import d2731 from './stimuli/dynamics/273_1t.mp4';
import d2822 from './stimuli/dynamics/282_2t.mp4';
import d2371 from './stimuli/dynamics/237_1t.mp4';
import "./styles/base.scss";

// function component() {
//   const element = document.createElement('div');
//
//   // Lodash, now imported by this script
//   element.innerHTML = _.join(['Hello', 'webpack'], ' ');
//
//   return element;
// }
//
// document.body.appendChild(component());



function MakeEventTimer() {
  this.t0 = new Date().getTime();
}

MakeEventTimer.prototype = {
  initialize() {
    this.t0 = new Date().getTime();
  },

  lap_reset() {
    let t1 = new Date().getTime() - this.t0;
    this.initialize();
    return t1 / 1000;
  },

  mark_diff() {
    let t1 = new Date().getTime() - this.t0;
    return t1 / 1000;
  }
}

function StimulusVideo(stimURL, videoPlayer, videoPlayer2, bypass_cors_, sandy_) {
  bypass_cors_ = bypass_cors_ || false;
  sandy_ = sandy_ || true;
  this.bypass_cors = !!bypass_cors_ && !!sandy_;
  this.stimURL = stimURL;
  this.stimFileName = stimURL.split('/').reverse()[0];
  this.videoPlayer = videoPlayer;
  this.videoPlayer2 = videoPlayer2 || false;
  this.blob = null;
  this.load_started = false;
  this.loaded = false;
  this.load_progress_elem = false;
  this.serve_requested = false;
  this.served = false;
  this.next_load_called = false;
  this.post_serve_called = false;
  this.disposed = false;
  this.active = false;
  this.time_to_load = 0;
  this.time_to_serve = 0;
  this.load_timer = new MakeEventTimer();
  this.serve_timer = new MakeEventTimer();
  this.sandy = sandy_;
  this.live_turkey = false;
  this.next_load = () => {
    console.log('terminal video preload')
  };
  this.next_serve = () => {
    console.log('disposing terminal: ')
  };
  this.post_serve = (stimfname = 'unlabeled') => {
    console.log('post serve: ', stimfname)
  };
  this.serve_requested_not_loaded_fn = () => {
    console.log('serve_requested_not_loaded_fn')
  };
}

StimulusVideo.prototype = {
  report() {
    return {
      stimFileName: this.stimFileName,
      time_to_load: this.time_to_load,
      time_to_serve: this.time_to_serve,
      active: this.active,
      served: this.served,
      disposed: this.disposed
    };
  },
  set_loaded(loaded = true) {
    this.loaded = loaded;
    this.time_to_load = this.load_timer.mark_diff();
    // console.log('LOADED this.stimURL: ', this.stimURL);
  },
  manage(disp_text = 'manage()') {

    if (!!this.disposed) {
      console.log('----- ERROR : stim already disposed -------');
      return 'stim already disposed';
    }

    if (!!this.loaded && !this.next_load_called) {
      this.next_load();
      this.next_load_called = true;
    }
    if (!!this.loaded && !!this.serve_requested && !this.served) {
      if (!!this.bypass_cors) {
        console.log('bypassing cors: ');
        this.videoPlayer.src = this.stimURL;
        if (!!this.videoPlayer2) {
          this.videoPlayer2.src = this.stimURL;
        }
      } else {
        this.videoPlayer.src = URL.createObjectURL(this.blob);
        if (!!this.videoPlayer2) {
          this.videoPlayer2.src = this.videoPlayer.src;
        }
      }
      this.served = true;
      this.active = true;
      this.time_to_serve = this.serve_timer.mark_diff();
      // if (this.sandy === true && this.live_turkey === false) {
      //   console.log('...-------vvvv serving');
      //   console.log('this.stimURL        : ', this.stimURL);
      //   console.log('this.videoPlayer.src: ', this.videoPlayer.src);
      //   console.log('this.time_to_serve: ', this.serve_timer.mark_diff());
      //   console.log('...-------^^^^ served');
      // }
    } else {
      // if (this.sandy === true && this.live_turkey === false) {
      //   console.log('-------vvvv NOT serving');
      //   console.log('not served:: ', this.stimURL);
      //   console.log('this.loaded: ', this.loaded);
      //   console.log('this.serve_requested: ', this.serve_requested);
      //   console.log('this.served: ', this.served);
      //   console.log('called from: ', disp_text);
      //   console.log('-------^^^^');
      // }

      if (!this.loaded && !!this.serve_requested && !this.served) {
        this.serve_requested_not_loaded_fn();
      }
    }
    if (!!this.loaded && !!this.serve_requested && !this.post_serve_called) {
      this.post_serve(this.stimFileName);
      this.post_serve_called = true;
    }
  },
  serve() {
    this.serve_requested = true;
    this.serve_timer.initialize();
    this.manage("serve -- ");
    // if (this.sandy === true && this.live_turkey === false) {
    //   console.log('===Serve requested: ', this.stimFileName);
    // }
  },
  load_direct() {
    let here = this;
    let timeout = !!this.sandy ? 2000 : 100;

    if (here.load_progress_elem !== false) {
      here.load_progress_elem.textContent = Math.floor(0.25 * 100) + '%';
    }
    setTimeout(() => {
      if (this.sandy === true && this.live_turkey === false) {
        console.log("BYPASSING BLOB: ", here.stimURL);
      }
      if (here.load_progress_elem !== false) {
        here.load_progress_elem.textContent = Math.floor(1.0 * 100).toString() + '%';
      }
      here.set_loaded();
      if (this.sandy === true && this.live_turkey === false) {
        console.log('====finished direct load of: ', here.stimURL);
      }
      here.manage("cors bypass loaded ::: ====finished load of: " + here.stimURL);
    }, timeout);
  },
  load_blob() {
    let here = this;

    let req = new XMLHttpRequest();
    req.open('GET', here.stimURL, true);
    req.responseType = 'blob';

    req.onload = function () {
      // Onload is triggered even on 404
      // so we need to check the status code
      if (this.status === 200) {
        here.blob = this.response;
        here.set_loaded();
        here.manage(".....>>>>>> XML loaded ::: ====finished load of: " + here.stimURL);
      } else {
        if (this.sandy === true && this.live_turkey === false) {
          console.log('====WARNING: this.status not 200, is: ', this.status);
        }
        here.bypass_cors = true;
        here.load_direct();
      }
    };
    req.onerror = () => {
      console.log("Booo - loading direct");
      here.bypass_cors = true;
      here.load_direct();
    };
    req.onprogress = (oEvent) => {
      if (here.load_progress_elem !== false && oEvent.lengthComputable) {
        let percentComplete = oEvent.loaded / oEvent.total;
        here.load_progress_elem.textContent = Math.floor(percentComplete * 100).toString() + '%';
      }
    }
    req.send();
  },
  load(load_progress_elem = false) {
    if (!!this.load_started) {
      if (this.sandy === true && this.live_turkey === false) {
        console.log('video load already initiated');
      }
    } else {
      this.load_started = true;
      if (load_progress_elem !== false) {
        this.load_progress_elem = load_progress_elem;
      }
      // if (this.sandy === true && this.live_turkey === false) {
      //   console.log('====starting load of: (this.bypass_cors = ', this.bypass_cors, ' ): ', this.stimURL);
      // }
      if (!!this.bypass_cors) {
        this.load_direct();
      } else {
        this.load_blob();
      }
      this.load_timer.initialize();
    }
  },
  dispose() {
    this.videoPlayer.src = URL.revokeObjectURL(this.videoPlayer.src), undefined;
    if (!!this.videoPlayer2) {
      this.videoPlayer2.src = this.videoPlayer.src;
    }
    this.blob = null;
    this.next_serve();
    this.active = false;
    this.disposed = true;
  },
  set_next_load(next_load) {
    this.next_load = next_load;
    this.next_load_called = false;
    this.manage("set_next_load");
  },
  set_next_serve(next_serve) {
    this.next_serve = next_serve;
  },
  set_post_serve(post_serve) {
    this.post_serve = post_serve;
    this.post_serve_called = false;
    this.manage("set_post_serve");
  },
  set_serve_requested_not_loaded_fn(serve_requested_not_loaded_fn) {
    this.serve_requested_not_loaded_fn = serve_requested_not_loaded_fn;
  }
}

const callback = function(){

  img_summaryfig
  img_splash

  // console.log('vid_training: ', vid_training);
  // let trainingVideo = new StimulusVideo(vid_training, document.getElementById("videoStim_training"));
  // trainingVideo.set_post_serve((stimulusfname = 'unlabeled') => {
  //   // document.getElementById("videoLoadingDiv").style.display = 'none';
  //   // document.getElementById("loadingTextLeft_training").style.visibility = 'hidden';
  //   // document.getElementById("loadingTextRight_training").style.visibility = 'hidden';
  //   //
  //   // document.getElementById("videoStim_training").addEventListener('ended', () => {
  //   //   maintaskParam.training_video_finished = true;
  //   // }, false);
  //   //
  //   // elements_named.training_video_button.disabled = false;
  //   console.log('training set_post_serve');
  // })
  // trainingVideo.load(document.getElementById("trainingVideoProgress"));
  // trainingVideo.serve();


  let player1 = new StimulusVideo(d2761, document.getElementById("videoStim_1"));
  player1.set_post_serve((stimulusfname = 'unlabeled') => {
  })
  player1.load();
  player1.serve();

  let player2 = new StimulusVideo(d2731, document.getElementById("videoStim_2"));
  player2.set_post_serve((stimulusfname = 'unlabeled') => {
  })
  player2.load();
  player2.serve();

  let player3 = new StimulusVideo(d2822, document.getElementById("videoStim_3"));
  player3.set_post_serve((stimulusfname = 'unlabeled') => {
  })
  player3.load();
  player3.serve();

  let player4 = new StimulusVideo(d2371, document.getElementById("videoStim_4"));
  player4.set_post_serve((stimulusfname = 'unlabeled') => {
  })
  player4.load();
  player4.serve();
};

if (
    document.readyState === "complete" ||
    (document.readyState !== "loading" && !document.documentElement.doScroll)
) {
  callback();
} else {
  document.addEventListener("DOMContentLoaded", callback);
}