import {mmtofurkey} from './mmtofurkey/mmtofurkey';
import $ from 'jquery';
import "./styles/styles1.scss";
// import "./styles/emotion_sliders_styles_less.less";
import "./styles/emotion_sliders_styles_scss.scss";

import {get_stim_param, get_stimulus_sets} from './stimuli_param';
import {detect_browser} from './detect_browser';

///////////////////////////////////////////////////////////////////

let sandy = false;
let bypass_cors = false;
let live_turkey = true;

var mmtofurkeyGravy = {
	// updateRandCondTable: function(rawData_floorLevel, turk) {

	// rawData_floorLevel.randC.....
	// },

	doNothing: function (rawData_floorLevel, turk) {
		// http://railsrescue.com/blog/2015-05-28-step-by-step-setup-to-send-form-data-to-google-sheets/
		try {
			console.log("happy eating");
		} catch (e) {
			console.log("tofurkeyGravy Error:", e);
		}
	}
};

var maintask;

const shuffledEmotionLabels = shuffleArray(["Annoyed", "Apprehensive", "Contemptuous", "Content", "Devastated", "Disappointed", "Disgusted", "Embarrassed", "Excited", "Furious", "Grateful", "Guilty", "Hopeful", "Impressed", "Jealous", "Joyful", "Proud", "Relieved", "Surprised", "Terrified"], false);

/////////////

var slidedisplay = {
	allSlides: document.querySelectorAll(".slideDiv"),
	validationSlides: [],
	validationSlidesShown: -1,
	after: "undefined",

	update_slide_deck() {
		this.allSlides = document.querySelectorAll(".slideDiv");
	},
	focus(slideId, scroll = true) {
		let found = false;
		for (let slide of this.allSlides) {
			if (slide.id === slideId) {
				replaceClass(slide, "slide-hidden", "slide-focused");
				found = true;
			} else {
				replaceClass(slide, "slide-focused", "slide-hidden");
			}
		}
		if (found === false) {
			console.log('====ERROR Slide NOT found. slideId : ', slideId);
			if (sandy === true && live_turkey === false) {
				console.log('this.allSlides: ', this.allSlides);
				alert('Slide Not Found. slideId: ' + slideId);
			}
		} else {
			if (scroll === true) {
				window.scrollTo(0, 0);
				// document.body.scrollTop = document.documentElement.scrollTop = 0;
			}
		}
	},
	next_validation_slide() {
		if (this.validationSlidesShown < this.validationSlides.length - 1) {
			console.log('Focusing Validation Slide (', this.validationSlidesShown + 1, '): ', this.validationSlides[this.validationSlidesShown + 1].id);
			this.focus(this.validationSlides[++this.validationSlidesShown].id);
		} else {
			console.log('Focusing Slide Final');
			this.after();
		}
	},
	set_validation_slides(id_list) {
		this.validationSlides = new Array(id_list.length);
		for (let ii = 0; ii < id_list.length; ii++) {
			this.validationSlides[ii] = document.getElementById(id_list[ii]);
		}
	},
	set_terminal_fn(final_fn) {
		this.after = final_fn;
	},
	get_validation_number() {
		return this.validationSlides.length;
	}
}

function replaceClass(elem, removeClass, addClass) {
	if (elem.classList.contains(removeClass)) {
		elem.classList.replace(removeClass, addClass);
	} else {
		if (elem.classList.contains(removeClass)) {
			elem.classList.remove(removeClass);
		}
		if (!elem.classList.contains(addClass)) {
			elem.classList.add(addClass);
		}
	}
}

function ResponseControl(query_gridworld, resp_set_beliefs, advanceRespButton, previousRespButton) {
	this.query_gridworld = document.getElementById(query_gridworld);
	this.resp_set = document.getElementById(resp_set_beliefs);
	this.button = document.getElementById(advanceRespButton);
	this.buttonBack = document.getElementById(previousRespButton);
	this.next = false;
	this.previous = false;
	this.respVerificationCallback = () => {
	};
	this.SuccessCallback = false;
	this.button.addEventListener('click', e => {
		e.currentTarget.blur();
		let respVarStatus = this.respVerificationCallback.check();
		if (respVarStatus === true) {
			this.hide();
			if (this.next) {
				this.next.show();
			}
			if (this.SuccessCallback) {
				this.SuccessCallback();
			}
			// window.scrollTo(0,0);
			document.body.scrollTop = document.documentElement.scrollTop = 0;
		}
	});
	this.buttonBack.addEventListener('click', e => {
		e.currentTarget.blur();
		if (this.previous) {
			this.hide();
			this.previous.show();
		}
	});
}

ResponseControl.prototype = {

	show() {
		this.resp_set.style.display = '';
		this.resp_set.style.visibility = 'visible';

		this.query_gridworld.style.display = '';
		this.query_gridworld.style.visibility = 'visible';
	},

	hide() {
		this.resp_set.style.display = 'none';
		this.resp_set.style.visibility = 'hidden';

		this.query_gridworld.style.display = 'none';
		this.query_gridworld.style.visibility = 'hidden';
	},

	init(next = () => {
	}, previous = () => {
	}, respVerificationCallback = () => {
	}, SuccessCallback = () => {
	}) {
		this.next = next;
		this.previous = previous;
		this.respVerificationCallback = respVerificationCallback;
		this.SuccessCallback = SuccessCallback;
		this.hide();
		if (!this.previous) {
			this.buttonBack.disabled = true;
			this.buttonBack.style.display = 'none';
			this.buttonBack.style.visibility = 'hidden';
		} else {
			this.buttonBack.childNodes[0].nodeValue = " << ";
		}
		if (!this.next) {
			this.button.childNodes[0].nodeValue = "Next Student";
		} else {
			this.button.childNodes[0].nodeValue = " >> ";
		}
	}
}


function ResponseControlVideo(videoPlayer, videoPlayerSmall, videoStimPackageDiv, videoStimFrame, button_play) {
	this.videoPlayer = videoPlayer;
	this.videoPlayerSmall = videoPlayerSmall;
	this.videoStimPackageDiv = videoStimPackageDiv;
	this.videoStimFrame = videoStimFrame;
	this.button_play = button_play;
	this.loading_text_left = document.getElementById("loadingTextLeft");
	this.loading_text_right = document.getElementById("loadingTextRight");
	this.pre_view_info = document.getElementById("responseDivCues1");
	this.n_presentations = 0;
	this.presentResponses = () => {
		console.log('present responses');
	};
}

ResponseControlVideo.prototype = {

	init() {
		this.button_play.addEventListener('click', e => {
			e.currentTarget.blur();
			this.playVideo();
		});
		this.videoPlayer.addEventListener('ended', () => {
			this.decisionPoint();
		}, false);
	},

	disable_play() {
		this.button_play.disabled = true;
		this.videoStimPackageDiv.style.opacity = "0.0";
		this.button_play.style.visibility = "visible";
		this.loading_text_left.style.visibility = "visible";
		this.loading_text_right.style.visibility = "visible";
	},

	prep() {
		this.videoStimPackageDiv.style.opacity = "0.0";

		this.n_presentations = 0;

		this.button_play.disabled = false;
		this.button_play.style.visibility = "visible";
		this.loading_text_left.style.visibility = "hidden";
		this.loading_text_right.style.visibility = "hidden";

		this.videoPlayer.pause();
		this.videoPlayer.currentTime = 0;

		this.videoPlayerSmall.pause();
		this.videoPlayerSmall.currentTime = 5.04;
		this.videoPlayerSmall.play();

		replaceClass(this.videoStimFrame, 'videoframe-running', 'videoframe-paused');
	},

	// on load, play button active, load text invisible, video opacity = 0
	// play button clicked, "Jackpot text removed", play button removed
	// 1s black
	// 1s on first frame
	// play video and change frame to white
	// on end, frame grey
	// 1s on last frame
	// 1s black
	// 1s first frame
	// play video and change frame to white
	// on end, frame grey
	// 1s on last frame
	// switch to responses

	decisionPoint() {
		if (sandy === true && live_turkey === false) {
			console.log('decisionPoint this.n_presentations: ', this.n_presentations);
		}

		if (this.n_presentations > 1) {
			if (sandy === true && live_turkey === false) {
				console.log('--- finalPresentation() ');
			}
			this.finalPresentation();
		} else {
			if (sandy === true && live_turkey === false) {
				console.log('--- secondPresentation() ', this.videoPlayer.currentTime);
			}
			this.secondPresentation();
		}
	},

	playVideo() {
		if (sandy === true && live_turkey === false) {
			console.log('___PLAYING___0: this.n_presentations: ', this.n_presentations);
		}
		this.button_play.disabled = true;
		this.n_presentations++;
		this.videoPlayer.pause();
		this.videoPlayer.currentTime = 0.0;
		this.button_play.style.visibility = "hidden";

		let here = this;
		setTimeout(() => { // 1s black
			// pause on first frame
			this.pre_view_info.style.display = 'none';
			here.videoStimPackageDiv.style.opacity = "1.0";
			setTimeout(() => {
				// play video and change frame to white
				if (sandy === true && live_turkey === false) {
					console.log('running');
				}

				replaceClass(here.videoStimFrame, 'videoframe-paused', 'videoframe-running');

				here.videoPlayer.play();

				if (sandy === true && live_turkey === false) {
					console.log('___PLAYING___1: this.n_presentati}ons: ', here.n_presentations);
				}

			}, 1000);
		}, 1000);
	},

	secondPresentation() {
		// on end, frame grey

		replaceClass(this.videoStimFrame, 'videoframe-running', 'videoframe-paused');

		let here = this;
		setTimeout(() => { // 1s on last frame
			// 1s black
			here.videoStimPackageDiv.style.opacity = "0.0";
			setTimeout(() => { // 1s on last frame
				here.playVideo();
			}, 1000);
		}, 1000);
	},

	finalPresentation() {
		// on end, frame grey
		replaceClass(this.videoStimFrame, 'videoframe-running', 'videoframe-paused');

		let here_presentResponses = this.presentResponses;
		setTimeout(() => { // 1s on last frame
			here_presentResponses();
		}, 1000);
	},

	set_presentResponses(presentResponses) {
		this.presentResponses = presentResponses;
		this.pre_view_info.style.display = '';
	}
}


function StimulusVideo(stimURL, videoPlayer, videoPlayer2) {
	this.bypass_cors = !!bypass_cors && !!sandy;
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
		if (live_turkey === false) {
			console.log('LOADED this.stimFileName: ', this.stimFileName);
		}
		if (sandy === true && live_turkey === false) {
			console.log('LOADED this.stimURL: ', this.stimURL);
		}
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
			if (sandy === true && live_turkey === false) {
				console.log('...-------vvvv serving');
				console.log('this.stimURL        : ', this.stimURL);
				console.log('this.videoPlayer.src: ', this.videoPlayer.src);
				console.log('this.time_to_serve: ', this.serve_timer.mark_diff());
				console.log('...-------^^^^ served');
			}
		} else {
			if (sandy === true && live_turkey === false) {
				console.log('-------vvvv NOT serving');
				console.log('not served:: ', this.stimURL);
				console.log('this.loaded: ', this.loaded);
				console.log('this.serve_requested: ', this.serve_requested);
				console.log('this.served: ', this.served);
				console.log('called from: ', disp_text);
				console.log('-------^^^^');
			}

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
		if (sandy === true && live_turkey === false) {
			console.log('===Serve requested: ', this.stimFileName);
		}
	},
	load_direct() {
		let here = this;
		let timeout = !!sandy ? 2000 : 100;

		if (here.load_progress_elem !== false) {
			here.load_progress_elem.textContent = Math.floor(0.25 * 100) + '%';
		}
		setTimeout(() => {
			if (sandy === true && live_turkey === false) {
				console.log("BYPASSING BLOB: ", here.stimURL);
			}
			if (here.load_progress_elem !== false) {
				here.load_progress_elem.textContent = Math.floor(1.0 * 100).toString() + '%';
			}
			here.set_loaded();
			if (sandy === true && live_turkey === false) {
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
				if (sandy === true && live_turkey === false) {
					console.log('====WARNING: this.status not 200, is: ', this.status);
				}
				here.bypass_cors = true;
				here.load_direct();
			}
		};
		req.onerror = () => {
			console.log("Booo");
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
			if (sandy === true && live_turkey === false) {
				console.log('video load already initiated');
			}
		} else {
			this.load_started = true;
			if (load_progress_elem !== false) {
				this.load_progress_elem = load_progress_elem;
			}
			if (sandy === true && live_turkey === false) {
				console.log('====starting load of: (this.bypass_cors = ', this.bypass_cors, ' ): ', this.stimURL);
			}
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

function lazyLoadImage(imageName, img) {
	import(
		/* webpackMode: "lazy-once" */
		`./images/${imageName}`
		)
		.then(src => img.src = src.default)
		.catch(err => console.error(err));
}

function validateResponses(responses, expected) {
	let pass = true;
	if (responses.length !== expected.length) {
		pass = false;
		// console.log('valid size mismatch ' + responses.length + '  ' + expected.length)
	} else {
		for (let i = 0; i < expected.length; i++) {
			if (expected[i] !== 'bypass') {
				if (responses[i] !== expected[i]) {
					pass = false;
					// console.log(responses[i] + ' vs ' + expected[i]);
				}
			}
		}
	}
	return pass;
}

function numberWithCommas(number) {
	return number.toFixed(2).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ","); // insert commas in thousands places in numbers with less than three decimals
	// return number.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ","); // insert commas in thousands places in numbers with less than three decimals
}

function getRadioResponse(radioName) {
	let radios = document.getElementsByName(radioName);
	for (let i = 0; i < radios.length; i++) {
		if (radios[i].checked === true) {
			return radios[i].value;
		}
	}
	return '';
}

function checkForUnansweredRadio(radioName) {
	let radios = document.getElementsByName(radioName);
	for (let i = 0; i < radios.length; i++) {
		if (radios[i].checked === true) {
			return true;
		}
	}
	return false;
}

function resetRadios(radioName) {
	let radios = document.getElementsByName(radioName);
	radios.forEach(e => {
		e.checked = false;
	})
}

function genIntRange(size, startAt = 0) {
	return [...Array(size).keys()].map(i => i + startAt);
}

function randomChoice(array) {
	return array[Math.floor(Math.random() * array.length)];
}

function shuffleArrayInplace(array) {
	/* Randomize array in-place using Durstenfeld shuffle algorithm */
	for (let i = array.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1));
		[array[i], array[j]] = [array[j], array[i]];
	}
}

function shuffleArray(array_in, inplace = true) {
	/* Randomize array in-place using Durstenfeld shuffle algorithm */
	if (!!inplace) {
		shuffleArrayInplace(array_in);
	} else {
		let new_array = [...array_in];
		shuffleArrayInplace(new_array);
		return new_array;
	}
}

function GenericRespVarFn(radios_to_check, debug = false) {
	this.radios_to_check = radios_to_check;
	this.sandy = debug;
}

GenericRespVarFn.prototype = {
	check() {
		let missing_resp = [];
		for (let ii = 0; ii < this.radios_to_check.length; ii++) {
			if (checkForUnansweredRadio(this.radios_to_check[ii]) !== true) {
				missing_resp.push(this.radios_to_check[ii]);
			}
		}
		if (missing_resp.length > 0) {
			console.log('missing_resp: ', missing_resp);
			if (this.sandy === false) {
				this.presentAlert(missing_resp);
				return false;
			} else {
				return true;
			}
		} else {
			return true;
		}
	},

	presentAlert(missing) {
		alert("Please respond to all questions.");
	}
}

function generateEmotionRatingRow(table, range_obj, div_obj, emolabel, training = false) {

	let emoLabelDisplay = document.getElementById('respRightEDisplayText');

	let row = table.insertRow();

	let cell1 = row.insertCell();
	let floor_anchor = document.createElement("SPAN");
	floor_anchor.append(document.createTextNode("not at all "));
	floor_anchor.classList.add("eFloor");
	cell1.appendChild(floor_anchor);

	let cell2 = row.insertCell();
	let slider_div = document.createElement("DIV");
	slider_div.classList.add("slider");
	slider_div.classList.add("label-untouched");

	let slider_label = document.createElement("LABEL");
	slider_label.classList.add("range-emotion-label");
	slider_label.appendChild(document.createTextNode(emolabel));

	let inputrange = document.createElement("INPUT");
	inputrange.classList.add("not-touched");
	inputrange.setAttribute("type", "range");
	inputrange.setAttribute("min", "0");
	inputrange.setAttribute("max", "48");
	inputrange.setAttribute("value", "0");

	slider_div.appendChild(slider_label);
	slider_div.appendChild(inputrange);
	cell2.appendChild(slider_div);

	let cell3 = row.insertCell();
	let ceiling_anchor = document.createElement("SPAN");
	ceiling_anchor.append(document.createTextNode(" extremely"));
	ceiling_anchor.classList.add("eCeiling");
	cell3.appendChild(ceiling_anchor);

	slider_div.classList.add("emotionRatingCell");
	cell1.classList.add("emotionAnchorCell");
	cell2.classList.add("emotionRatingCellSliderContainer");
	cell3.classList.add("emotionAnchorCell");

	floor_anchor.style.visibility = 'hidden';
	ceiling_anchor.style.visibility = 'hidden';

	let touch_row = () => {
		if (inputrange.classList.contains("not-touched")) {
			inputrange.classList.remove("not-touched");
		}
		if (slider_div.classList.contains("label-untouched")) {
			slider_div.classList.remove("label-untouched");
		}
		if (!slider_div.classList.contains("label-touched")) {
			slider_div.classList.add("label-touched");
		}
	};

	let floor_range = () => {
		inputrange.value = "0";
		touch_row();
	};
	let ceil_range = () => {
		inputrange.value = "48";
		touch_row();
	};

	cell1.addEventListener('click', e => floor_range());
	cell3.addEventListener('click', e => ceil_range());

	inputrange.addEventListener('click', e => touch_row());
	inputrange.addEventListener('mousedown', e => touch_row());

	if (training === true) { /// training table
		row.onmouseenter = (e) => {
			floor_anchor.style.visibility = 'visible';
			ceiling_anchor.style.visibility = 'visible';
		}
		row.onmouseleave = (e) => {
			floor_anchor.style.visibility = 'hidden';
			ceiling_anchor.style.visibility = 'hidden';
		}
	} else { /// not training table
		row.onmouseenter = (e) => {
			floor_anchor.style.visibility = 'visible';
			ceiling_anchor.style.visibility = 'visible';
			emoLabelDisplay.textContent = emolabel;
		}
		row.onmouseleave = (e) => {
			floor_anchor.style.visibility = 'hidden';
			ceiling_anchor.style.visibility = 'hidden';
			emoLabelDisplay.textContent = ' ';
		}
	}

	range_obj[emolabel] = inputrange;
	div_obj[emolabel] = slider_div;
}


function generateValidationRatingRow(table) {

	let row = table.insertRow(1);

	let cell1 = row.insertCell();
	cell1.classList.add("textRight");
	let floor_anchor = document.createElement("SPAN");
	floor_anchor.append(document.createTextNode("Split "));
	floor_anchor.classList.add("eFloorLarger");
	cell1.appendChild(floor_anchor);

	let cell2 = row.insertCell();
	let slider_div = document.createElement("DIV");
	slider_div.classList.add("slider");
	slider_div.classList.add("label-untouched");

	let slider_label = document.createElement("LABEL");
	slider_label.classList.add("range-emotion-label");
	slider_label.appendChild(document.createTextNode("Split \u00A0 \u00A0 \u00A0 Steal"));

	let inputrange = document.createElement("INPUT");
	inputrange.classList.add("not-touched");
	inputrange.setAttribute("type", "range");
	inputrange.setAttribute("min", "0");
	inputrange.setAttribute("max", "100");
	inputrange.setAttribute("value", "0");

	slider_div.appendChild(slider_label);
	slider_div.appendChild(inputrange);
	cell2.appendChild(slider_div);

	let cell3 = row.insertCell();
	cell3.classList.add("textRight");
	let ceiling_anchor = document.createElement("SPAN");
	ceiling_anchor.append(document.createTextNode(" Steal"));
	ceiling_anchor.classList.add("eCeilingLarger");
	cell3.appendChild(ceiling_anchor);

	slider_div.classList.add("emotionRatingCell");
	cell1.classList.add("emotionAnchorCell");
	cell2.classList.add("emotionRatingCellSliderContainer");
	cell3.classList.add("emotionAnchorCell");

	floor_anchor.style.visibility = 'visible';
	ceiling_anchor.style.visibility = 'visible';

	let touch_row = () => {
		if (inputrange.classList.contains("not-touched")) {
			inputrange.classList.remove("not-touched");
		}
		if (slider_div.classList.contains("label-untouched")) {
			slider_div.classList.remove("label-untouched");
		}
		if (!slider_div.classList.contains("label-touched")) {
			slider_div.classList.add("label-touched");
		}
	};

	let predicted_C_proport = document.getElementById("predicted_C_proport");
	let predicted_D_proport = document.getElementById("predicted_D_proport");
	let update_anchor_text = () => {
		let val = inputrange.value;
		predicted_C_proport.textContent = (100 - Number(val)).toString();
		predicted_D_proport.textContent = val;
	}

	let floor_range = () => {
		inputrange.value = "0";
		touch_row();
		update_anchor_text();
	};
	let ceil_range = () => {
		inputrange.value = "100";
		touch_row();
		update_anchor_text();
	};

	cell1.addEventListener('click', e => floor_range());
	cell3.addEventListener('click', e => ceil_range());

	inputrange.addEventListener('click', e => touch_row());
	inputrange.addEventListener('mousedown', e => touch_row());

	inputrange.addEventListener('input', e => {
		update_anchor_text();
	});

	return [inputrange, slider_div];
}

let call_condition = {
	baseURL: 'https://daeda.scripts.mit.edu/serveCondition/serveCondition.php?callback=?',
	condfname: "servedConditions_exp12.csv",
	writestatus: "TRUE",
	n_conditions: 132,
	succeeded: 'undefined',
	request_time: 0.0,
	reply_result: 'undefined',
	reply_time: 0.0,

	requestService: '',
	res: {condNum: "undefined"},

	set_write_status(write_status = true) {
		if (write_status === false) {
			let php_var = "FALSE";
			this.writestatus = php_var;
			console.log('PHP write status set successfully to: ', this.writestatus);
			return php_var;
		} else {
			let php_var = "TRUE";
			this.writestatus = php_var;
			console.log('PHP write status set successfully to: ', this.writestatus);
			return php_var;
		}
	},

	set_res(res) {
		this.res = res;
	},

	determine_write_status(assignmentId, turkSubmitTo) {

		// If there's no turk info
		if (!assignmentId || !turkSubmitTo) {
			console.log("Dead Turkey: Not writing to conditions file.");
			console.log("    assignmentId = ", assignmentId);
			console.log("    turkSubmitTo_local = ", turkSubmitTo);
			live_turkey = false;
			return this.set_write_status(false);
		} else {
			console.log("Live Turkey!");
			sandy = false;
			bypass_cors = false;
			live_turkey = true;
			return this.set_write_status(true);
		}
	},

	gen_condNum() {
		let n_conditions = 132;
		return randomChoice(genIntRange(n_conditions)).toString();
	},
	get_condition_number() {
		let i_condition = parseInt(this.res.condNum);
		if (i_condition >= 0 && i_condition < this.n_conditions) {
			return i_condition.toString();
		} else {
			if (sandy === true && live_turkey === false) {
				let timeout = 1500;
				setTimeout(() => {
					console.log('===ERROR, this.res.condNum not recognized: ', this.res.condNum);
				}, timeout);
			}
			return this.gen_condNum();
		}
	},

	get(turk, elements_named, show_slide_id = false) {
		let here = this;

		let request_timer = new MakeEventTimer();

		// determine if HIT is live
		let assignmentId = turk.assignmentId;
		let turkSubmitTo = turk.turkSubmitTo;

		let write_status = here.determine_write_status(assignmentId, turkSubmitTo)

		let requestService = 'writestatus=' + write_status + '&condComplete=' + 'REQUEST' + '&condfname=' + here.condfname;

		$.ajax({
			url: here.baseURL,
			dataType: "jsonp",
			type: "GET",
			data: requestService,
		})
			.done((data) => {
				here.set_res(data);
				here.succeeded = true;
				console.log("php success");
			})
			.fail(() => {
				here.succeeded = false;
				console.log("WARNING php failed");
			})
			.always(() => {
				console.log('Serve returned: ', here.res.condNum);
				let set_id = 'set' + here.get_condition_number();
				loadHIT(turk, elements_named, set_id, show_slide_id);
				here.request_time = request_timer.mark_diff();
			});

		here.requestService = requestService;
	},

	reply(randCond, subject_valid = true, on_done_fn = false) {
		let here = this;

		let reply_timer = new MakeEventTimer();

		let randCondNum = randCond.split('set')[1];

		let returnServe = 'writestatus=' + here.writestatus + '&condComplete=' + randCondNum + '&subjValid=' + subject_valid.toString().toUpperCase() + '&condfname=' + here.condfname;

		if (sandy === true && live_turkey === false) {
			console.log("attempting to return condition");
			console.log('php: ' + returnServe);
		}

		$.ajax({
			url: here.baseURL,
			dataType: "jsonp",
			type: "GET",
			data: returnServe,
		})
			.done((data) => {
				console.log("Serve Returned!", data.condNum);
				here.reply_result = true;
			})
			.fail(() => {
				console.log("Serve returned failed");
				here.reply_result = false;
			})
			.always(() => {
				here.reply_time = reply_timer.mark_diff();
				if (on_done_fn !== false) {
					// // Show the 'you can exit' slide
					// slidedisplay.focus(show_slide_id);
					on_done_fn(here.reply_result, here.reply_time);
				}
			});
	},
}

function loadHIT(turk, elements_named, set_id = "undefined", show_slide_id = false) {

	if (show_slide_id !== false) {
		slidedisplay.focus(show_slide_id);
	}

	//////////////////

	const [conditionId, selectedSet] = get_stimulus_sets(set_id);
	let selectedStimIds;
	if (sandy === true && live_turkey === false) {
		selectedStimIds = new Array(3);
		for (let stimid_idx = 0; stimid_idx < 3; stimid_idx++) {
			selectedStimIds[stimid_idx] = 'stimid' + selectedSet[stimid_idx];
		}
	} else {
		selectedStimIds = new Array(selectedSet.length);
		for (let stimid_idx = 0; stimid_idx < selectedSet.length; stimid_idx++) {
			selectedStimIds[stimid_idx] = 'stimid' + selectedSet[stimid_idx];
		}
	}
	if (sandy === true && live_turkey === false) {
		console.log('--selectedSet (' + selectedSet.length + '): ', selectedSet);
		console.log('--selectedStimIds (' + selectedStimIds.length + '): ', selectedStimIds);
	}

	let maintaskParam = new SetMaintaskParam(selectedStimIds);


	/// populate trial numbers
	elements_named.total_num_questions.forEach(e => {
		e.textContent = maintaskParam.numTrials.toString()
	});
	slidedisplay.set_validation_slides(elements_named.validation_slides);

	const videoStim = document.getElementById("videoStim");
	const videoStimSmall = document.getElementById("videoStim_small");

	const delms = {
		interaction_mask: document.getElementById('interactionMask'),
	};

	const flowctrl = {
		video_control: new ResponseControlVideo(videoStim, videoStimSmall, document.getElementById("videoStimPackageDiv"), document.getElementById("videoStimFrameID"), elements_named.button_play),
	}
	flowctrl.video_control.init();


	let serverRoot = '../../';
	// this.stimNum = stimNum;
	// this.stimURL = serverRoot + "dynamics/" + maintaskParam.allTrialOrders[ maintaskParam.shuffledOrder[stimNum] ].stimulus + "t.mp4";

	document.getElementById("payoff_comprehension_CD12_photothumb_p1").src = serverRoot + "stimuli/statics/" + "238_1" + ".png";
	document.getElementById("payoff_comprehension_DD42_photothumb_p1_1").src = serverRoot + "stimuli/statics/" + "236_1" + ".png";
	document.getElementById("payoff_comprehension_DD42_photothumb_p1_2").src = serverRoot + "stimuli/statics/" + "236_1" + ".png";
	// document.getElementById("payoff_comprehension_CD12_photothumb").src = serverRoot + "stimuli/statics/" + "generic_avatar_male.png" + ".png";

	elements_named.training_video_button.disabled = true;
	let trainingVideo = new StimulusVideo(serverRoot + "stimuli/dynamics/" + "258_c_ed_vbr2" + ".mp4", document.getElementById("videoStim_training"));
	trainingVideo.set_post_serve((stimulusfname = 'unlabeled') => {
		document.getElementById("videoLoadingDiv").style.display = 'none';
		document.getElementById("loadingTextLeft_training").style.visibility = 'hidden';
		document.getElementById("loadingTextRight_training").style.visibility = 'hidden';

		document.getElementById("videoStim_training").addEventListener('ended', () => {
			maintaskParam.training_video_finished = true;
		}, false);

		elements_named.training_video_button.disabled = false;
	})
	trainingVideo.load(document.getElementById("trainingVideoProgress"));
	trainingVideo.serve();

	// document.getElementById("revoke_button").addEventListener('click', e => {
	//     e.currentTarget.blur();
	//     trainingVideo.dispose();
	// });
	elements_named.trainingVideo = trainingVideo;


	elements_named.video_objects = {};
	for (let istim = 0; istim < maintaskParam.subsetStimOrdered.length; istim++) {
		// https://developer.mozilla.org/en-US/docs/Learn/JavaScript/Client-side_web_APIs/Video_and_audio_APIs
		// https://www.w3schools.com/tags/ref_av_dom.asp
		// let videoelem = document.createElement("VIDEO");
		// videoelem.setAttribute('')
		// videoContainerSmall.appendChild(videoelem);
		// crossOrigin
		// controls
		let vidId = maintaskParam.subsetStimOrdered[istim].stimulus;
		// let stimURL = serverRoot + "stimuli/dynamics/" + maintaskParam.allTrialOrders[ maintaskParam.shuffledOrder[stimNum] ].stimulus + "t.mp4";
		let stimURL = serverRoot + "stimuli/dynamics/" + vidId + "t.mp4";
		let video_obj = new StimulusVideo(stimURL, videoStimSmall, videoStim);

		video_obj.set_post_serve((stimulusfname = 'unlabeled') => {
			console.log('>>>> postserve stimulusfname: ', stimulusfname);
			flowctrl.video_control.prep();
			maintaskParam.visible_video_base = vidId;
			maintaskParam.visible_video = stimulusfname;
		})

		video_obj.set_serve_requested_not_loaded_fn(() => {
			flowctrl.video_control.disable_play();
		})

		if (istim === 0) {
			/// if first stim video, serve immediately
			trainingVideo.set_next_load(() => {
				if (live_turkey === false) {
					console.log('Starting Stim Load: ');
				}
				video_obj.load();
				video_obj.serve();
			});
		} else {
			let preceedingVid = maintaskParam.subsetStimOrdered[istim - 1].stimulus;
			elements_named.video_objects[preceedingVid].set_next_load(() => {
				if (live_turkey === false) {
					console.log('preloading : ', vidId, "  (", istim, ")");
				}
				video_obj.load();
			});
			elements_named.video_objects[preceedingVid].set_next_serve(() => {
				if (live_turkey === false) {
					console.log('serving : ', vidId, "  (", istim, ")");
				}
				video_obj.serve();
			});
		}

		elements_named.video_objects[vidId] = video_obj;
	}

	let timer_exp = new MakeEventTimer();
	let timer_trial = new MakeEventTimer();

	let progressBarObj = {
		progress_tracker_bar: document.querySelector(".bar"),
		progress_tracker_totalnum: document.getElementById("total-num"),
		progress_tracker_trialnum: document.getElementById("trial-num"),
		numTrials: 0,
		numValidationQuestions: 0,
		barSize: 300.0,

		update(numComplete = -1) { // Update progress bar
			let totalTrials = this.numTrials + this.numValidationQuestions;
			let trialNum = Math.min(Math.max(0, numComplete + 1), totalTrials);
			this.progress_tracker_bar.style.width = Math.round(this.barSize * trialNum / totalTrials).toString() + "px";
			this.progress_tracker_trialnum.textContent = trialNum.toString();
			this.progress_tracker_totalnum.textContent = (totalTrials).toString();
		},

		add_trial(num_to_add = 1) {
			this.numTrials = this.numTrials + num_to_add;
		},

		add_validation(num_to_add = 1) {
			this.numValidationQuestions = this.numValidationQuestions + num_to_add;
		},

	};
	progressBarObj.add_trial(maintaskParam.numTrials);
	// progressBarObj.add_validation(slidedisplay.get_validation_number());

	maintask = {

		emotion_order: shuffledEmotionLabels,
		randCondNum: conditionId,

		validationRadio: {},
		dem_gender: "",
		dem_language: "",
		val_recognized: "",
		val_familiar: "",
		val_feedback: "",
		training_video_stats: {},

		data: [],
		dataInSitu: [],

		total_time: 0,
		visible_area: [document.documentElement.clientWidth, document.documentElement.clientHeight],
		browser: detect_browser(),
		browser_version: navigator.userAgent,
		ip: "none",
		ipify_calls: 0,
		request_succeeded: 'none',
		request_time: 0,
		reply_succeeded: 'none',
		reply_time: 0,

		set_ip(ip) {
			this.ip = ip;
			if (sandy === true) {
				console.log('ip set to: ', this.ip);
			}
		},

		ipify_called() {
			this.ipify_calls++;
		},

		set_reply(result = 'none', time = 0) {
			this.reply_succeeded = result;
			this.reply_time = time;
		},

		issue_reply() {
			let randCond = this.randCondNum;
			let subject_valid = true;
			let call_on_reply_fn = (res, time) => {
				maintask.set_reply(res, time);
			};
			call_condition.reply(randCond, subject_valid, call_on_reply_fn);
			console.log('this.reply_succeeded: ', this.reply_succeeded);
			console.log('this.reply_time: ', this.reply_time);
		},

		finalForcedQuestions(fieldnames_in, nextslide) {
			let fieldnames = Array.isArray(fieldnames_in) === false ? [fieldnames_in] : fieldnames_in;

			/// check for missing values
			let missing_resp = false;

			for (let fieldname of fieldnames) {
				if (getRadioResponse(fieldname) === '') {
					missing_resp = true;
					console.log('NOT FOUND : ', fieldname);
				}
			}

			if (missing_resp === false) { /// if all good
				for (let fieldname of fieldnames) {
					this.validationRadio[fieldname] = getRadioResponse(fieldname);
				}
				slidedisplay.focus(nextslide);
			} else {
				alert("Please answer all of the questions");
				if (sandy === true && live_turkey === false) {
					console.log('bypassing check: finalForcedQuestions');
					for (let fieldname of fieldnames) {
						this.validationRadio[fieldname] = getRadioResponse(fieldname);
					}
					slidedisplay.focus(nextslide);
				}
			}

			if (sandy === true && live_turkey === false) {
				console.log('this.validationRadio: ', this.validationRadio);
			}

			return !missing_resp;
		},


		finalForcedQuestionsRange(range, label, nextslide) {

			if (!!range.classList.contains("not-touched")) {
				alert("Please respond to the question.");
				if (sandy === true) {
					this.validationRadio[label] = Number(range.value);
					slidedisplay.focus(nextslide);
				}
			} else {
				this.validationRadio[label] = Number(range.value);
				slidedisplay.focus(nextslide);
			}

			if (sandy === true) { console.log('this.validationRadio: ', this.validationRadio); }

		},

		check_if_advance_allowed() {
			let unanswered = [];
			let any_unanswered = false;
			for (let emo of this.emotion_order) {
				if (!!elements_named.emo_ranges[emo].classList.contains("not-touched")) {
					any_unanswered = true;
					unanswered.push(emo);
				}
			}
			return [any_unanswered, unanswered];
		},

		end() {
			// stop experiment timer
			this.total_time = timer_exp.mark_diff();
			console.log('---this.total_time: ', this.total_time);

			this.dem_gender = getRadioResponse("dem_gender");
			this.dem_language = document.getElementById("dem_language").value;
			this.val_recognized = document.getElementById("val_recognized").value;
			this.val_familiar = document.getElementById("val_familiar").value;
			this.val_feedback = document.getElementById("val_feedback").value;

			this.request_time = call_condition.request_time;
			this.request_succeeded = call_condition.succeeded;

			this.issue_reply();

			// SEND DATA TO TURK
			setTimeout(() => {
				turk.submit(maintask, true, mmtofurkeyGravy);
				setTimeout(() => {
					// Show the 'you can exit' slide
					slidedisplay.focus("exit");
				}, 1000); // time to wait on "finished" page after submitting maintask
			}, 2000); // time to wait on "finished" page before submitting maintask

			// Show the finished slide ('results being submitted').
			slidedisplay.focus("finished");
		},

		store() {
			let trial = {};

			// record stimulus param
			trial.trial_number = maintaskParam.numComplete;
			trial.stimParam = JSON.stringify(maintaskParam.stimParam);
			trial.visible_video = maintaskParam.trialControl.visible_video.slice();
			trial.visible_video_base = maintaskParam.trialControl.visible_video_base.slice();
			trial.video_stats = JSON.parse(JSON.stringify(elements_named.video_objects[maintaskParam.stimParam.stimulus].report()));
			trial.visible_area = [document.documentElement.clientWidth, document.documentElement.clientHeight];

			// record response time
			trial.respTimer = timer_trial.lap_reset();

			// record response data
			for (let emo of this.emotion_order) {
				trial[emo] = Number(elements_named.emo_ranges[emo].value);
			}

			if (sandy === true) { console.log('trial: ', trial); }

			/// store
			this.data.push(trial);
		},

		reset() {
			for (let emo of this.emotion_order) {
				elements_named.emo_ranges[emo].value = "0";
				elements_named.emo_ranges[emo].classList.add("not-touched");
				elements_named.slider_divs[emo].classList.remove("label-touched");
			}
			// maintaskParam.trial = {};
			// maintaskParam.trialControl = {};
			maintaskParam.stimParam = {};
		},

		next() {

			// present_responses_button.disabled = true;
			// advance_frame_button.disabled = false;

			// duplicate allConditionStim
			if (maintaskParam.numComplete < 0) { // if this is the first trial
				timer_trial.initialize();
				let debugScan = true;
				maintaskParam.debugScan = sandy === true ? debugScan : false;
				elements_named.trainingVideo.dispose();
				this.training_video_stats = JSON.parse(JSON.stringify(elements_named.trainingVideo.report()));
			}

			// If this is not the first trial, record variables
			console.log('maintaskParam.numComplete: ', maintaskParam.numComplete);
			if (maintaskParam.numComplete >= 0) { // If this isn't the first time .next() has been called
				/// display next video
				elements_named.video_objects[maintaskParam.stimParam.stimulus].dispose();

				/// store
				this.store();

				/// reset
				this.reset();
			}

			// If subject has completed all trials, update progress bar and
			// show slide to ask for demographic info
			if (maintaskParam.numComplete >= maintaskParam.numTrials - 1) { // If all the trials have been presented
				slidedisplay.next_validation_slide();

				// Update progress bar
				progressBarObj.update(maintaskParam.numComplete);

				maintaskParam.all_trials_finished = true;

				// Otherwise, if trials not completed yet, update progress bar
				// and go to next trial based on the order in which trials are supposed
				// to occur

			} else { // If there are more trials to present

				/// advance stimulus

				maintaskParam.stimParam = maintaskParam.subsetStimOrdered[++maintaskParam.numComplete];
				// maintaskParam.trialControl = {};

				// slidedisplay.focus("transitionMask");
				// setTimeout(() => {
				//     slidedisplay.focus("slideStimulus");
				//     window.scrollTo(0,0);
				// }, 500);

				elements_named.stimulus_pot_span.textContent = "Jackpot: $" + numberWithCommas(maintaskParam.stimParam.pot) + " ";
				elements_named.response_pot_span.textContent = "$" + numberWithCommas(maintaskParam.stimParam.pot) + " ";

				slidedisplay.focus("slideStimulus");

				// maintaskParam.advanceStim = false;

				//// agent
				// elements_named.name.forEach(e => {e.textContent = maintaskParam.stimParam.agent_name });

				// Update progress bar
				progressBarObj.update(maintaskParam.numComplete);

			}
			// let startTime = (new Date()).getTime();
			// let endTime = (new Date()).getTime();
			//key = (keyCode == 80) ? "p" : "q",
			//userParity = experiment.keyBindings[key],
			// data = {
			//   stimulus: n,
			//   accuracy: realParity == userParity ? 1 : 0,
			//   rt: endTime - startTime
			// };

			// experiment.data.push(data);
			//setTimeout(experiment.next, 500);

			// window.scrollTo(0,0);
			// delms.interaction_mask.style.display = 'none';
			timer_trial.initialize();
		}

	};

	flowctrl.video_control.set_presentResponses(() => {
		slidedisplay.focus("slideResponse");
		maintaskParam.trialControl.visible_video = maintaskParam.visible_video.slice();
		maintaskParam.trialControl.visible_video_base = maintaskParam.visible_video_base.slice();
	});

	slidedisplay.set_terminal_fn(maintask.end);

	elements_named.button_next.addEventListener('click', e => {
		e.currentTarget.blur();
		let next_check = maintask.check_if_advance_allowed();
		if (!!next_check[0]) {
			alert("Please respond to all questions.");
			if (sandy === true) {
				maintask.next();
			}
		} else {
			maintask.next();
		}
	});

	elements_named.training_video_button.addEventListener('click', e => {
		e.currentTarget.blur();
		let field_names = "val_trainingvideo";
		let next_slide = "training4";
		if (!!maintaskParam.training_video_finished) {
			if (!!maintask.finalForcedQuestions(field_names, next_slide)) {
				/// if questions pass, dispose of video
				elements_named.trainingVideo.dispose();
			}
		} else {
			alert("Please watch the entire video and answer the question below.");

			if (sandy === true) {
				console.log('bypassing check: training video');
				maintask.finalForcedQuestions(field_names, next_slide);
			}
		}
	});

	document.getElementById("final_button").addEventListener('click', e => {
		e.currentTarget.blur();
		maintask.end()
	});

	document.getElementById("nextTraining_buttonLast").addEventListener('click', e => {
		e.currentTarget.blur();
		maintask.next();
	});

	document.getElementById("nextConcluding_button1").addEventListener('click', e => {
		e.currentTarget.blur();
		maintask.finalForcedQuestions('val_emomatch_contemptuous', 'concluding2');
	});

	document.getElementById("nextConcluding_button2").addEventListener('click', e => {
		e.currentTarget.blur();
		maintask.finalForcedQuestions('val_expressionmatch_joyful', 'concluding3');
	});

	document.getElementById("nextConcluding_button3").addEventListener('click', e => {
		e.currentTarget.blur();
		maintask.finalForcedQuestionsRange(elements_named.a1_range_validation, 'predicted_C_proport', 'concluding4');
	});

	document.getElementById("nextConcluding_button4").addEventListener('click', e => {
		e.currentTarget.blur();
		maintask.finalForcedQuestions(['iwould_large', 'iexpectother_large'], 'concluding5');
	});

	document.getElementById("nextConcluding_button5").addEventListener('click', e => {
		e.currentTarget.blur();
		maintask.finalForcedQuestions(['payoff_comprehension_CD12_p1', 'payoff_comprehension_CD12_p2'], 'concluding6');
	});

	document.getElementById("nextConcluding_button6").addEventListener('click', e => {
		e.currentTarget.blur();
		maintask.finalForcedQuestions(['payoff_comprehension_DD42_p1', 'payoff_comprehension_CC42_p1'], 'final');
	});

	elements_named.fist_training_advance_button.addEventListener('click', e => {
		e.currentTarget.blur();

		let next_slide = "training2";

		let unanswered = [];
		let any_unanswered = false;
		for (let emo of Object.getOwnPropertyNames(elements_named.emo_ranges_training)) {
			if (!!elements_named.emo_ranges_training[emo].classList.contains("not-touched")) {
				any_unanswered = true;
				unanswered.push(emo);
			}
		}

		if (unanswered.length > 0) {
			alert("Please provide an answer to all emotions.");
		} else if (Number(elements_named.emo_ranges_training["Apprehensive"].value) !== 0) {
			alert("The sliding range of --apprehensive-- is not set to the minimum possible value. Please click the --not at all-- text to the left of the grey bar.");
		} else if (Number(elements_named.emo_ranges_training["Excited"].value) < 10 || Number(elements_named.emo_ranges_training["Excited"].value) > 38) {
			alert("The sliding range of --excited-- is not set near the mid-point of the grey bar. Please move the marker towards the middle.");
		} else if (Number(elements_named.emo_ranges_training["Furious"].value) !== 48) {
			alert("The sliding range of --furious-- is not set all the way to the maximum possible value. Please click and drag the marker all the way to the right of the grey bar.");
		} else { // all checks pass
			slidedisplay.focus(next_slide);
		}

		if (sandy === true) {
			console.log('sandy : bypassing check');
			slidedisplay.focus(next_slide);
		}

	});

	document.getElementById("nextTraining_button2").addEventListener('click', e => {
		e.currentTarget.blur();
		slidedisplay.focus("slideTrainingVideo");
	});

	document.getElementById("nextTraining_button4").addEventListener('click', e => {
		e.currentTarget.blur();
		slidedisplay.focus("training5");
	});

	(function ipifysearch(param) {
		$.ajax({
			url: "https://api.ipify.org?format=json",
			dataType: 'json',
			// data: data,
			timeout: 4000 // 4 second timeout
		})
			.done((data) => {
				console.log("ipify success");
				maintask.set_ip(data.ip);
			})
			.fail(() => {
				console.log("WARNING ipify failed");
				let retryAfter = param.retryAfter;
				if (param.active === true && maintaskParam.all_trials_finished === false) { /// keep trying while these conditions are met
					setTimeout(() => {
						ipifysearch(param)
					}, retryAfter);
				} else {
					maintask.set_ip('maxout');
				}
			})
			.always(() => {
				let display_ipify = sandy === true ? maintask.ip : 'ipify finished';
				maintask.ipify_called();
				console.log('display_ipify: ', display_ipify);
			});
	})({retryAfter: 7000, active: true});

}


///////////////////////////////////////////////////////////////////


function SetMaintaskParam(selectedTrialIds) {

	this.selectedTrialIds = shuffleArray(selectedTrialIds, false);
	this.numTrials = this.selectedTrialIds.length;
	this.subsetStimOrdered = new Array(this.numTrials);
	for (let stimid_idx = 0; stimid_idx < this.numTrials; stimid_idx++) {
		this.subsetStimOrdered[stimid_idx] = get_stim_param(this.selectedTrialIds[stimid_idx]);
	}
	this.numComplete = -1;
	this.storeDataInSitu = false;

	this.visible_video = 'unset';
	this.visible_video_base = 'unset';

	this.trialControl = {};

	this.training_video_finished = false;

	this.all_trials_finished = false;

	if (sandy === true && live_turkey === false) {
		console.log('this.subsetStimOrdered (' + this.numTrials.toString() + ') :  ', this.subsetStimOrdered);
	}
}


document.addEventListener('DOMContentLoaded', () => {

	// const img = document.createElement('img');
	// src="img/MIT-logo-with-spelling-office-red-gray-design4.png" alt="MIT" width="584" height="80"
	// const logoimg = document.getElementById('logo');
	// lazyLoadImage(BoxImg, logoimg);
	// logoimg.src = BoxImg;
	// mitlogo

	let turk;
	turk = turk || {};
	mmtofurkey(turk, window, document);

	slidedisplay.update_slide_deck();
	slidedisplay.focus("splash");

	console.log('===sandy: ', sandy);

	document.body.addEventListener('oncontextmenu', e => {
		return false;
	})

	////////

	document.getElementById("allEmoLabelsStr").textContent = shuffledEmotionLabels.join(", ");

	let emo_ranges = {};
	let slider_divs = {};
	shuffledEmotionLabels.forEach(emolabel => generateEmotionRatingRow(document.getElementById("responsesTable"), emo_ranges, slider_divs, emolabel));

	let emo_ranges_training = {};
	let slider_divs_training = {};
	generateEmotionRatingRow(document.getElementById("responsesTableTraining1"), emo_ranges_training, slider_divs_training, "Apprehensive", true);
	generateEmotionRatingRow(document.getElementById("responsesTableTraining2"), emo_ranges_training, slider_divs_training, "Excited", true);
	generateEmotionRatingRow(document.getElementById("responsesTableTraining3"), emo_ranges_training, slider_divs_training, "Furious", true);

	let a1_range_validation, a1_slider_divs_validation;
	[a1_range_validation, a1_slider_divs_validation] = generateValidationRatingRow(document.getElementById("responsesTableValidation"));

	//////////

	const elementsNamed = {
		emo_ranges: emo_ranges,
		slider_divs: slider_divs,
		emo_ranges_training: emo_ranges_training,
		a1_range_validation: a1_range_validation,
		button_next: document.getElementById("next_stim"),
		button_play: document.getElementById("playButton"),
		response_pot_span: document.getElementById("response_pot_span"),
		stimulus_pot_span: document.getElementById("stimulus_pot_span"),
		total_num_questions: Array.from(document.getElementsByClassName("numTrials")),
		training_video_button: document.getElementById("nextTraining_button_video"),
		fist_training_advance_button: document.getElementById("nextTraining_button1"),
		validation_slides: ["concluding1", "concluding2", "final"],
	};
	document.getElementById("payoff_display").style.display = 'none';


	document.getElementById("begin_button").addEventListener('click', e => {
		e.currentTarget.blur();
		if (!!turk.previewMode) {
			alert("Please accept this HIT to see more questions.");
		} else {
			slidedisplay.focus("loadingHIT");
			call_condition.get(turk, elementsNamed, "training1");
		}
	});

	////////////////////// DEBUGGING ////////////////////

	if (sandy === true) {
		console.log('====DEBUG=====');

		// document.getElementById("TEMP_start_button").addEventListener('click', e => {
		//     e.currentTarget.blur();
		//     // maintask.next();
		//     // trainingVideo.dispose();
		//     // slidedisplay.focus('slideResponse');
		//     maintask.next();
		// });

		// const tempendbutton = document.getElementById("TEMP_end_exp");
		// tempendbutton.addEventListener('click', e => {e.currentTarget.blur(); maintask.end()});

	}
});