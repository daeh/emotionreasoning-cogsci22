///////////// head /////////////

/// initalize sliders ///
$('.not-clicked').mousedown(function() {
	$(this).removeClass('not-clicked').addClass('slider-input');
	$(this).closest('.slider').children('.slider-label-not-clicked').removeClass('slider-label-not-clicked').addClass('slider-label');
});

/// initalize rollover effects of emotion labels ///
$('.emotionRatingRow').hover(
	function() {
		$('#respRightEDisplayText').html($(this).children('td').children('div.slider').children('.isCenter').text());
		$(this).closest("tr").children("td").children("span.eFloor").html('not&nbsp;at&nbsp;all');
		$(this).closest("tr").children("td").children("span.eFloor").closest("td").click(function() {
			$(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val('0');
			$(this).closest("tr").children("td").children("div.slider").children('.slider-label-not-clicked').removeClass('slider-label-not-clicked').addClass('slider-label');
			$(this).closest("tr").children("td").children("div.slider").children('.not-clicked').removeClass('not-clicked').addClass('slider-input');
			$("#barChart").height(Math.round((getResponseDivSizes() / 48) * $(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val()));
		});
		$(this).closest("tr").children("td").children("span.eCeiling").html('extremely');
		$(this).closest("tr").children("td").children("span.eCeiling").closest("td").click(function() {
			$(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val('100');
			$(this).closest("tr").children("td").children("div.slider").children('.slider-label-not-clicked').removeClass('slider-label-not-clicked').addClass('slider-label');
			$(this).closest("tr").children("td").children("div.slider").children('.not-clicked').removeClass('not-clicked').addClass('slider-input');
			$("#barChart").height(Math.round((getResponseDivSizes() / 48) * $(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val()));
		});
		$("#barChart").height(Math.round((getResponseDivSizes() / 48) * $(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val()));
	},
	function() {
		$('#respRightEDisplayText').html('&nbsp;');
		$(this).closest("tr").children("td").children("span.eFloor").html('&nbsp;');
		$(this).closest("tr").children("td").children("span.eCeiling").html('&nbsp;');
		$("#barChart").height(0);
	}
);

// $('.emotionRatingRow').mouseup(function() {$('#TEMP').html( $(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val() ) } );
// $('.emotionRatingRow').mousemove(function() { $('#TEMP').html($(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val()) });
$('.emotionRatingRow').mouseup(function() {
	$("#barChart").height(
		Math.round((getResponseDivSizes() / 48) * $(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val())
	);
});
$('.emotionRatingRow').mousemove(function() {
	$("#barChart").height(
		Math.round((getResponseDivSizes() / 48) * $(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val())
	);
});


///////////// Window /////////////


/// MTURK Initalization ///

var iSlide = 0;
var iTrial = 0;
var serverRoot = "../stimuli/"; // requires terminal slash
var serverRoot1 = ""; // requires terminal slash

var subjectValid = true;


/// UX Control ///

function interferenceHandler(e) {
	e.stopPropagation();
	e.preventDefault();
}

function checkPreview() {
	if (!!turk.previewMode) {
		alert("Please accept this HIT to see more questions.");
		return false;
	}
	return true;
}


/// User Input Control ///

// Radio Buttons //
function getRadioCheckedValue(formNum, radio_name) {
	var oRadio = document.forms[formNum].elements[radio_name];
	for (var i = 0; i < oRadio.length; i++) {
		if (oRadio[i].checked) {
			return oRadio[i].value;
		}
	}
	return '';
}

// Ranges //
function ValidateRanges() {
	var pass = true;
	var unanswered = $("#responsesTable").find(".slider-label-not-clicked");
	if (unanswered.length > 0) {
		pass = false;
		alert("Please provide an answer to all emotions. If you think that a person is not experiencing a given emotion, rate that emotion as --not at all-- by moving the sliding marker all the way to the left.");
	}
	return pass;
}

function validateDemoRanges() {
	var pass = true;

	var unanswered = $("#demoTable").find(".slider-label-not-clicked");
	if (unanswered.length > 0) {
		pass = false;
		alert("Please provide an answer to all emotions.");
	} else if (demo1.value != 0) {
		pass = false;
		alert("The sliding range of --apprehensive-- is not set to the minimum possible value. Please click the --not at all-- text to the left of the grey bar.");
	} else if (demo2.value < 10 || demo2.value > 38) {
		pass = false;
		alert("The sliding range of --excited-- is not set near the mid-point of the grey bar. Please move the marker towards the middle.");
	} else if (demo3.value != 48) {
		pass = false;
		alert("The sliding range of --furious-- is not set all the way to the maximum possible value. Please click and drag the marker all the way to the right of the grey bar.");
	}
	return pass;
}

function ResetRanges() {
	var ranges = $('input[type="range"]');
	for (var i = 0; i < ranges.length; i++) {
		ranges[i].value = "0";
	}
	ranges.removeClass('slider-input').addClass('not-clicked');
	ranges.closest('.slider').children('label').removeClass("slider-label").addClass("slider-label-not-clicked");
}

// Textareas //
function ValidateFieldEquivalence(testField, targetString) {
	return ValidateTextEquivalence(testField.value, targetString);
}

function ValidateTextEquivalence(test, target) {
	var valid = true;
	var parsedTarget = parseWords(target);
	var parsedField = parseWords(test);
	if (!(parsedTarget.length === parsedField.length)) {
		valid = false;
	} else {
		for (var i = 0; i < parsedTarget.length; i++) {
			if (!(parsedTarget[i].toUpperCase() === parsedField[i].toUpperCase())) {
				valid = false;
			}

		}
	}
	return valid;
}

function ValidateText(field, min, max) {
	var valid = true;

	if (field.value === "") {
		alert("Please provide an answer.");
		valid = false;
	} else {
		var values = parseWords(field.value);
		if (values.length > max || values.lengt < min) {
			// invalid word number
			return false;
		}
	}
	return valid;
}

function parseWords(string) {
	// !variable will be true for any falsy value: '', 0, false, null, undefined. null == undefined is true, but null === undefined is false. Thus string == null, will catch both undefined and null.
	// (typeof string === 'undefined' || !string)
	if (!string) {
		var values = "";
	} else {
		var values = string.replace(/\n/g, " ").split(' ').filter(function(v) {
			return v !== '';
		});
	}
	return values;
}


/// Genertic Functions ///

function genIntRange(min, max) {
	var range = [];
	for (var i = min; i <= max; i++) {
		range.push(i);
	}
	return range;
}

/**
 * Randomize array element order in-place.
 * Using Durstenfeld shuffle algorithm.
 */
function shuffleArray(array) {
	for (var i = array.length - 1; i > 0; i--) {
		var j = Math.floor(Math.random() * (i + 1));
		var temp = array[i];
		array[i] = array[j];
		array[j] = temp;
	}
	return array;
}

/* 
random(a,b)
Returns random number between a and b, inclusive
*/
function random(a, b) {
	if (typeof b == "undefined") {
		a = a || 2;
		return Math.floor(Math.random() * a);
	} else {
		return Math.floor(Math.random() * (b - a + 1)) + a;
	}
}

/* 
Array.prototype.random
Randomly shuffles elements in an array. Useful for condition randomization.

E.G.
var choices = ["rock", "paper", "scissors"];
var computer = { "play":choices.random() };

console.log("The computer picked " + computer["play"]+ ".");
*/
Array.prototype.random = function() {
	return this[random(this.length)];
}


/// Window Control ///

function slideUp() {
	iSlide++;
}

function showSlide(id) {
	$(".slide").hide();
	$("#" + id).show();
	$("#responseDivCues0").show();
}

function presentStim() {
	if (nPresentations === 0) {
		smallVideo.src = document.getElementById("videoStim").src;
		setTimeout(smallVideo.pause(), 300);
		setTimeout(function() { smallVideo.currentTime = 5.04 }, 1000);

		document.addEventListener("clicks", interferenceHandler, true);
		document.addEventListener("contextmenu", interferenceHandler, true);
		$('#interactionMask').show();
		$('#videoStimPackageDiv').css('opacity', '0');
		$("#videoStimPackageDiv").show();
		disablePlayButton();
		$("#playButtonContainer").hide();
		playVid();
	}
}


function presentResponses() {
	showSlide("slideResponse");
	// getResponseDivSizes();

	$("#contextTableFrontDiv").html('&nbsp;');

	if (maintaskParam.numComplete <= 1) {
		setResponseTableSizes();
	}
	$(".responseBarChartBackground").height(getResponseDivSizes());

	// $('#outer_responseCollection').show();

	// $("#responseTableFrame").height($('#responseTableFrameID').outerHeight());

	// $('#outer_stimPresentation').hide();
	// $('#outer_stimPresentation').addClass('element_purge'); // totally overkill could mess with alignment

}

function setResponseTableSizes() {
	// console.log('$("#responseTableFrameID").height()', $("#responseTableFrameID").height());
	// console.log('$("#responseCuesTableFrameID").height()', $("#responseCuesTableFrameID").height());
	// console.log('',);

	// var minTableHeight = Math.max($("#responseTableFrameID").height(), $("#responseCuesTableFrameID").height());

	// $("#responseTableFrameID").height(minTableHeight)
	// $("#responseCuesTableFrameID").height(minTableHeight);

	var tableHeight = getResponseDivSizes(); // (30 is cell padding + border)*2  //This works in chrome and safari, but makes the frame 30 too big in Firefox DEBUG

	// console.log('tableHeight', tableHeight);

	$("#responseTableFrameID").height(tableHeight);
	$("#responseCuesTableFrameID").height(tableHeight);

	// return minTableHeight;
}

function getResponseDivSizes() {
	return ($('#RespSet').outerHeight());
}

/// Stim Control ///

// var smallVideo = document.getElementById("videoStim_small");

// function setSmallVid()
// {
//     var promise = longfunctionfirst().then(shortfunctionsecond);
// }
// function longfunctionfirst()
// {
//     var d = new $.Deferred();
//     setTimeout(function() {
// 
// d.resolve();
//     },1);
//     return d.promise();
// }
// function shortfunctionsecond()
// {
//     var d = new $.Deferred();
//     setTimeout(function() {
//     	smallVideo.play();
//     	smallVideo.pause();
//     	smallVideo.currentTime = 5;
//     	d.resolve();
//     },10);
//     return d.promise();
// }

var thisVideo = document.getElementById("videoStim");
var nPresentations = 0;

function stimHandler() {
	document.getElementById('videoStim').removeEventListener('ended', stimHandler, false);
	setTimeout(function() {
		$('#videoStimPackageDiv').css('opacity', '0');
		if (nPresentations < 2) {
			setTimeout(playVid, 1000);
		} else {
			$("playButtonContainer").hide();
			document.removeEventListener("clicks", interferenceHandler, true);
			document.removeEventListener("contextmenu", interferenceHandler, true);
			$('#interactionMask').hide();

			presentResponses();
			preloadStim(maintaskParam.numComplete); // load next movie
		}
	}, 1000);
}

function playVid() {
	nPresentations++;
	thisVideo.pause();
	thisVideo.currentTime = 0;

	setTimeout(function() {
		$('#videoStimPackageDiv').css('opacity', '1');
		setTimeout(function() {
			thisVideo.play();
			document.getElementById('videoStim').addEventListener('ended', stimHandler, false);
		}, 1000);
	}, 1000);
}

/*
// FETCH calls not supported by legacy browsers (or currently iOS). Using XML instead for the time being.
// load next video
function preloadStim(stimNum) {
	// var tempurl = serverRoot + "dynamics/" + maintaskParam.allTrialOrders[maintaskParam.shuffledOrder[ stimNum ]].stimulus + "t.mp4";
	fetch(serverRoot + "dynamics/" + maintaskParam.allTrialOrders[maintaskParam.shuffledOrder[stimNum]].stimulus + "t.mp4").then(function(response) {
		return response.blob();
	}).then(function(data) {
		document.getElementById("videoStim").src = URL.createObjectURL(data);
		enablePlayButton();
		console.log("Current load: stimNum    ", stimNum, "maintaskParam.shuffledOrder[ stimNum ]                  ", maintaskParam.shuffledOrder[stimNum], "stimID", maintaskParam.allTrialOrders[maintaskParam.shuffledOrder[stimNum]].stimulus, "( maintaskParam.numComplete", maintaskParam.numComplete, ")");
		// console.log("maintaskParam.numComplete", maintaskParam.numComplete, "maintaskParam.shuffledOrder[ maintaskParam.numComplete ]", maintaskParam.shuffledOrder[ maintaskParam.numComplete ], "stimID", maintaskParam.allTrialOrders[maintaskParam.shuffledOrder[ maintaskParam.numComplete ]].stimulus);
	}).catch(function() {
		console.log("Booo");
	});
}
*/

function preloadStim(stimNum) {
	var req = new XMLHttpRequest();
	var stimURL = serverRoot + "dynamics/" + maintaskParam.allTrialOrders[maintaskParam.shuffledOrder[stimNum]].stimulus + "t.mp4";
	req.open('GET', stimURL, true);
	req.responseType = 'blob';

	req.onload = function() {
		// Onload is triggered even on 404
		// so we need to check the status code
		if (this.status === 200) {
			var videoBlob = this.response;
			document.getElementById("videoStim").src = URL.createObjectURL(videoBlob); // IE10+
			enablePlayButton();
			console.log("Current load: stimNum    ", stimNum, "maintaskParam.shuffledOrder[ stimNum ]                  ", maintaskParam.shuffledOrder[stimNum], "stimID", maintaskParam.allTrialOrders[maintaskParam.shuffledOrder[stimNum]].stimulus, "( maintaskParam.numComplete", maintaskParam.numComplete, ")");
		}
	};
	req.onerror = function() {
		console.log("Booo");
	};
	req.send();
}


// Play button control //
function disablePlayButton() {
	$('#playButton').prop('onclick', null).off('click');
	$('#playButton').removeClass('play-button').addClass('play-button-inactive');
	$("#loadingTextLeft").html('VIDEO&nbsp;');
	$("#loadingTextRight").html('&nbsp;LOADING');
}

function enablePlayButton() {
	$('#playButton').click(function() {
		presentStim();
		$("#responseDivCues0").hide();
	});
	$('#playButton').removeClass('play-button-inactive').addClass('play-button');
	$("#loadingTextLeft").html('&nbsp;');
	$("#loadingTextRight").html('&nbsp;');
	nPresentations = 0;
}

function numberWithCommas(number) {
	return number.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ","); // insert commas in thousands places in numbers with less than three decimals
	// return number.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ","); // insert commas in thousands places in numbers with less than three decimals
}


var mmtofurkeyGravy = {
	// updateRandCondTable: function(rawData_floorLevel, turk) {

	// rawData_floorLevel.randC.....
	// },

	doNothing: function(rawData_floorLevel, turk) {
		// http://railsrescue.com/blog/2015-05-28-step-by-step-setup-to-send-form-data-to-google-sheets/
		try {
			console.log("happy eating");
		} catch (e) { console.log("tofurkeyGravy Error:", e); }
	}
};

var trainingVideo = {


	preloadStim: function() {
		var req = new XMLHttpRequest();
		var stimURL = serverRoot + "dynamics/" + "258_c_ed_vbr2.mp4";
		req.open('GET', stimURL, true);
		req.responseType = 'blob';

		req.onload = function() {
			// Onload is triggered even on 404
			// so we need to check the status code
			if (this.status === 200) {
				var videoBlob = this.response;
				document.getElementById("videoStim_training").src = URL.createObjectURL(videoBlob); // IE10+
				// document.getElementById("videoStim_training").src = stimURL;
				console.log("Current load: training complete    ", document.getElementById("videoStim_training").src);
				// $('#playButton_training').click(function() {
				// 	presentStim();
				// });
				$("#loadingTextLeft_training").html('&nbsp;');
				$("#loadingTextRight_training").html('&nbsp;');
				$("#videoLoadingDiv").hide();
				// $("#videoStimPackageDiv_training").show()
				// console.log("Current load: stimNum    ", stimNum);

				document.getElementById('videoStim_training').addEventListener('ended', enableAdvance, false);
			}
		};
		req.onerror = function() {
			console.log("Booo");
		};
		req.send();
	},

}

function enableAdvance() {
	$('#training_advance_button').removeClass('advance-button-inactive').addClass('advance-button');
	$('#training_advance_button').removeClass('advance-button-inactive').addClass('advance-button');
	document.getElementById('training_advance_button').onclick = function() {
		if(!!maintask.validate0('7510')){this.blur(); showSlide(4)};

	}}

/// Experiment ///

function SetMaintaskParam(selectedTrials) {
	this.allConditions = [
		[
			{ "condition": 1, "Version": "1a", "stimID": 235.1, "stimulus": "235_1", "decisionThis": "Stole", "decisionOther": "Split", "pot": 50221.00 }, // 0
			{ "condition": 1, "Version": "1a", "stimID": 235.2, "stimulus": "235_2", "decisionThis": "Split", "decisionOther": "Stole", "pot": 50221.00 }, // 1
			{ "condition": 1, "Version": "1a", "stimID": 237.1, "stimulus": "237_1", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 65673.50 }, // 2
			{ "condition": 1, "Version": "1a", "stimID": 237.2, "stimulus": "237_2", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 65673.50 }, // 3
			{ "condition": 1, "Version": "1a", "stimID": 239.1, "stimulus": "239_1", "decisionThis": "Split", "decisionOther": "Split", "pot": 94819.00 }, // 4
			{ "condition": 1, "Version": "1a", "stimID": 239.2, "stimulus": "239_2", "decisionThis": "Split", "decisionOther": "Split", "pot": 94819.00 }, // 5
			{ "condition": 1, "Version": "1a", "stimID": 240.1, "stimulus": "240_1", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 36159.00 }, // 6
			{ "condition": 1, "Version": "1a", "stimID": 240.2, "stimulus": "240_2", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 36159.00 }, // 7
			{ "condition": 1, "Version": "1a", "stimID": 241.1, "stimulus": "241_1", "decisionThis": "Split", "decisionOther": "Split", "pot": 80954.50 }, // 8
			{ "condition": 1, "Version": "1a", "stimID": 241.2, "stimulus": "241_2", "decisionThis": "Split", "decisionOther": "Split", "pot": 80954.50 }, // 9
			{ "condition": 1, "Version": "1a", "stimID": 242.1, "stimulus": "242_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 1283.50 }, // 10
			{ "condition": 1, "Version": "1a", "stimID": 242.2, "stimulus": "242_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 1283.50 }, // 11
			{ "condition": 1, "Version": "1a", "stimID": 243.1, "stimulus": "243_1", "decisionThis": "Split", "decisionOther": "Split", "pot": 7726.50 }, // 12
			{ "condition": 1, "Version": "1a", "stimID": 243.2, "stimulus": "243_2", "decisionThis": "Split", "decisionOther": "Split", "pot": 7726.50 }, // 13
			{ "condition": 1, "Version": "1a", "stimID": 244.1, "stimulus": "244_1", "decisionThis": "Split", "decisionOther": "Split", "pot": 139.00 }, // 14
			{ "condition": 1, "Version": "1a", "stimID": 244.2, "stimulus": "244_2", "decisionThis": "Split", "decisionOther": "Split", "pot": 139.00 }, // 15
			{ "condition": 1, "Version": "1a", "stimID": 245.1, "stimulus": "245_1", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 5958.00 }, // 16
			{ "condition": 1, "Version": "1a", "stimID": 245.2, "stimulus": "245_2", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 5958.00 }, // 17
			{ "condition": 1, "Version": "1a", "stimID": 247.1, "stimulus": "247_1", "decisionThis": "Stole", "decisionOther": "Split", "pot": 3.50 }, // 18
			{ "condition": 1, "Version": "1a", "stimID": 247.2, "stimulus": "247_2", "decisionThis": "Split", "decisionOther": "Stole", "pot": 3.50 }, // 19
			{ "condition": 1, "Version": "1a", "stimID": 248.1, "stimulus": "248_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 31403.00 }, // 20
			{ "condition": 1, "Version": "1a", "stimID": 248.2, "stimulus": "248_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 31403.00 }, // 21
			{ "condition": 1, "Version": "1a", "stimID": 249.1, "stimulus": "249_1", "decisionThis": "Split", "decisionOther": "Split", "pot": 1562.50 }, // 22
			{ "condition": 1, "Version": "1a", "stimID": 249.2, "stimulus": "249_2", "decisionThis": "Split", "decisionOther": "Split", "pot": 1562.50 }, // 23
			{ "condition": 1, "Version": "1a", "stimID": 250.1, "stimulus": "250_1", "decisionThis": "Split", "decisionOther": "Split", "pot": 30304.50 }, // 24
			{ "condition": 1, "Version": "1a", "stimID": 250.2, "stimulus": "250_2", "decisionThis": "Split", "decisionOther": "Split", "pot": 30304.50 }, // 25
			{ "condition": 1, "Version": "1a", "stimID": 251.1, "stimulus": "251_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 2835.50 }, // 26
			{ "condition": 1, "Version": "1a", "stimID": 251.2, "stimulus": "251_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 2835.50 }, // 27
			{ "condition": 1, "Version": "1a", "stimID": 252.1, "stimulus": "252_1", "decisionThis": "Stole", "decisionOther": "Split", "pot": 310.00 }, // 28
			{ "condition": 1, "Version": "1a", "stimID": 252.2, "stimulus": "252_2", "decisionThis": "Split", "decisionOther": "Stole", "pot": 310.00 }, // 29
			{ "condition": 1, "Version": "1a", "stimID": 253.1, "stimulus": "253_1", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 21719.50 }, // 30
			{ "condition": 1, "Version": "1a", "stimID": 253.2, "stimulus": "253_2", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 21719.50 }, // 31
			{ "condition": 1, "Version": "1a", "stimID": 254.1, "stimulus": "254_1", "decisionThis": "Split", "decisionOther": "Split", "pot": 130884.00 }, // 32
			{ "condition": 1, "Version": "1a", "stimID": 254.2, "stimulus": "254_2", "decisionThis": "Split", "decisionOther": "Split", "pot": 130884.00 }, // 33
			{ "condition": 1, "Version": "1a", "stimID": 256.1, "stimulus": "256_1", "decisionThis": "Stole", "decisionOther": "Split", "pot": 2983.00 }, // 34
			{ "condition": 1, "Version": "1a", "stimID": 256.2, "stimulus": "256_2", "decisionThis": "Split", "decisionOther": "Stole", "pot": 2983.00 }, // 35
			{ "condition": 1, "Version": "1a", "stimID": 257.1, "stimulus": "257_1", "decisionThis": "Split", "decisionOther": "Split", "pot": 30.00 }, // 36
			{ "condition": 1, "Version": "1a", "stimID": 257.2, "stimulus": "257_2", "decisionThis": "Split", "decisionOther": "Split", "pot": 30.00 }, // 37
			{ "condition": 1, "Version": "1a", "stimID": 260.1, "stimulus": "260_1", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 1598.50 }, // 38
			{ "condition": 1, "Version": "1a", "stimID": 260.2, "stimulus": "260_2", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 1598.50 }, // 39
			{ "condition": 1, "Version": "1a", "stimID": 262.1, "stimulus": "262_1", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 56488.00 }, // 40
			{ "condition": 1, "Version": "1a", "stimID": 262.2, "stimulus": "262_2", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 56488.00 }, // 41
			{ "condition": 1, "Version": "1a", "stimID": 263.1, "stimulus": "263_1", "decisionThis": "Stole", "decisionOther": "Split", "pot": 28802.00 }, // 42
			{ "condition": 1, "Version": "1a", "stimID": 263.2, "stimulus": "263_2", "decisionThis": "Split", "decisionOther": "Stole", "pot": 28802.00 }, // 43
			{ "condition": 1, "Version": "1a", "stimID": 264.1, "stimulus": "264_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 24145.00 }, // 44
			{ "condition": 1, "Version": "1a", "stimID": 264.2, "stimulus": "264_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 24145.00 }, // 45
			{ "condition": 1, "Version": "1a", "stimID": 265.1, "stimulus": "265_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 1588.00 }, // 46
			{ "condition": 1, "Version": "1a", "stimID": 265.2, "stimulus": "265_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 1588.00 }, // 47
			{ "condition": 1, "Version": "1a", "stimID": 266.1, "stimulus": "266_1", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 6559.00 }, // 48
			{ "condition": 1, "Version": "1a", "stimID": 266.2, "stimulus": "266_2", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 6559.00 }, // 49
			{ "condition": 1, "Version": "1a", "stimID": 268.1, "stimulus": "268_1", "decisionThis": "Split", "decisionOther": "Split", "pot": 61381.50 }, // 50
			{ "condition": 1, "Version": "1a", "stimID": 268.2, "stimulus": "268_2", "decisionThis": "Split", "decisionOther": "Split", "pot": 61381.50 }, // 51
			{ "condition": 1, "Version": "1a", "stimID": 269.1, "stimulus": "269_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 2318.00 }, // 52
			{ "condition": 1, "Version": "1a", "stimID": 269.2, "stimulus": "269_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 2318.00 }, // 53
			{ "condition": 1, "Version": "1a", "stimID": 270.1, "stimulus": "270_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 5322.50 }, // 54
			{ "condition": 1, "Version": "1a", "stimID": 270.2, "stimulus": "270_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 5322.50 }, // 55
			{ "condition": 1, "Version": "1a", "stimID": 271.1, "stimulus": "271_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 19025.50 }, // 56
			{ "condition": 1, "Version": "1a", "stimID": 271.2, "stimulus": "271_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 19025.50 }, // 57
			{ "condition": 1, "Version": "1a", "stimID": 272.1, "stimulus": "272_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 81899.00 }, // 58
			{ "condition": 1, "Version": "1a", "stimID": 272.2, "stimulus": "272_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 81899.00 }, // 59
			{ "condition": 1, "Version": "1a", "stimID": 273.1, "stimulus": "273_1", "decisionThis": "Split", "decisionOther": "Split", "pot": 9675.00 }, // 60
			{ "condition": 1, "Version": "1a", "stimID": 273.2, "stimulus": "273_2", "decisionThis": "Split", "decisionOther": "Split", "pot": 9675.00 }, // 61
			{ "condition": 1, "Version": "1a", "stimID": 276.1, "stimulus": "276_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 130944.00 }, // 62
			{ "condition": 1, "Version": "1a", "stimID": 276.2, "stimulus": "276_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 130944.00 }, // 63
			{ "condition": 1, "Version": "1a", "stimID": 277.1, "stimulus": "277_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 3331.00 }, // 64
			{ "condition": 1, "Version": "1a", "stimID": 277.2, "stimulus": "277_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 3331.00 }, // 65
			{ "condition": 1, "Version": "1a", "stimID": 278.1, "stimulus": "278_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 40606.00 }, // 66
			{ "condition": 1, "Version": "1a", "stimID": 278.2, "stimulus": "278_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 40606.00 }, // 67
			{ "condition": 1, "Version": "1a", "stimID": 279.1, "stimulus": "279_1", "decisionThis": "Split", "decisionOther": "Split", "pot": 269.50 }, // 68
			{ "condition": 1, "Version": "1a", "stimID": 279.2, "stimulus": "279_2", "decisionThis": "Split", "decisionOther": "Split", "pot": 269.50 }, // 69
			{ "condition": 1, "Version": "1a", "stimID": 281.1, "stimulus": "281_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 1803.00 }, // 70
			{ "condition": 1, "Version": "1a", "stimID": 281.2, "stimulus": "281_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 1803.00 }, // 71
			{ "condition": 1, "Version": "1a", "stimID": 282.1, "stimulus": "282_1", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 2843.50 }, // 72
			{ "condition": 1, "Version": "1a", "stimID": 282.2, "stimulus": "282_2", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 2843.50 }, // 73
			{ "condition": 1, "Version": "1a", "stimID": 283.1, "stimulus": "283_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 57518.00 }, // 74
			{ "condition": 1, "Version": "1a", "stimID": 283.2, "stimulus": "283_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 57518.00 }, // 75
			{ "condition": 1, "Version": "1a", "stimID": 284.1, "stimulus": "284_1", "decisionThis": "Stole", "decisionOther": "Split", "pot": 1116.50 }, // 76
			{ "condition": 1, "Version": "1a", "stimID": 284.2, "stimulus": "284_2", "decisionThis": "Split", "decisionOther": "Stole", "pot": 1116.50 }, // 77
			{ "condition": 1, "Version": "1a", "stimID": 285.1, "stimulus": "285_1", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 2300.50 }, // 78
			{ "condition": 1, "Version": "1a", "stimID": 285.2, "stimulus": "285_2", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 2300.50 }, // 79
			{ "condition": 1, "Version": "1a", "stimID": 286.1, "stimulus": "286_1", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 3532.50 }, // 80
			{ "condition": 1, "Version": "1a", "stimID": 286.2, "stimulus": "286_2", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 3532.50 }, // 81
			{ "condition": 1, "Version": "1a", "stimID": 287.1, "stimulus": "287_1", "decisionThis": "Split", "decisionOther": "Stole", "pot": 1030.50 }, // 82
			{ "condition": 1, "Version": "1a", "stimID": 287.2, "stimulus": "287_2", "decisionThis": "Stole", "decisionOther": "Split", "pot": 1030.50 }, // 83
			{ "condition": 1, "Version": "1a", "stimID": 288.1, "stimulus": "288_1", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 15744.50 }, // 84
			{ "condition": 1, "Version": "1a", "stimID": 288.2, "stimulus": "288_2", "decisionThis": "Stole", "decisionOther": "Stole", "pot": 15744.50 }, // 85
			{ "condition": 1, "Version": "1a", "stimID": 289.1, "stimulus": "289_1", "decisionThis": "Stole", "decisionOther": "Split", "pot": 822.50 }, // 86
			{ "condition": 1, "Version": "1a", "stimID": 289.2, "stimulus": "289_2", "decisionThis": "Split", "decisionOther": "Stole", "pot": 822.50 } // 87
		]
	];

	/* Experimental Variables */
	// Number of conditions in experiment
	this.numConditions = 1; //allConditions.length;

	// Randomly select a condition number for this particular participant
	this.chooseCondition = 1; // random(0, numConditions-1);

	// Based on condition number, choose set of input (trials)
	this.allTrialOrders = this.allConditions[this.chooseCondition - 1];

	// Produce random order in which the trials will occur
	// this.shuffledOrder = shuffleArray(genIntRange(0, this.allTrialOrders.length - 1));
	this.shuffledOrder = shuffleArray(selectedTrials);
	console.log('shuffledOrder', this.shuffledOrder);

	// Number of trials in each condition
	this.numTrials = this.shuffledOrder.length; //not necessarily this.allTrialOrders.length;

	// Pull the random subet
	this.subsetTrialOrders = [];
	for (var i = 0; i < this.numTrials; i++) {
		this.subsetTrialOrders.push(this.allTrialOrders[i]);
	}

	// Keep track of current trial 
	this.currentTrialNum = 0;

	// Keep track of how many trials have been completed
	this.numComplete = 0;

	this.storeDataInSitu = false;
}

var phpParam = { baseURL: 'https://daeda.scripts.mit.edu/GB/serveCondition.php?callback=?', nReps: 1, condfname: "servedConditions.csv" };

function loadHIT(nextSlide) {
	// determine if HIT is live
	var assignmentId_local = turk.assignmentId,
		turkSubmitTo_local = turk.turkSubmitTo;

	// If there's no turk info
	phpParam.writestatus = "TRUE";
	if (!assignmentId_local || !turkSubmitTo_local) {
		console.log("Dead Turkey: Not writing to conditions file.");
		phpParam.writestatus = "FALSE";
	} else {
		console.log("Live Turkey!");
		phpParam.writestatus = "TRUE";
	}

	var requestService = 'nReps=' + phpParam.nReps + '&writestatus=' + phpParam.writestatus + '&condComplete=' + 'REQUEST' + '&condfname=' + phpParam.condfname;

	showSlide("loadingHIT");
	$.getJSON(phpParam.baseURL, requestService, function(res) {

		console.log("Served Condition:", res.condNum);

		showSlide(nextSlide);
		var defaultCondNum = 86; // if PHP runs out of options
		var conditions = [
			[21, 5, 2, 31, 39, 33, 54, 53, 44, 67, 68, 76], // 0
			[5, 2, 15, 39, 33, 34, 53, 44, 59, 68, 76, 73], // 1
			[2, 15, 10, 33, 34, 29, 44, 59, 61, 76, 73, 77], // 2
			[15, 10, 6, 34, 29, 32, 59, 61, 50, 73, 77, 71], // 3
			[10, 6, 16, 29, 32, 37, 61, 50, 57, 77, 71, 69], // 4
			[6, 16, 13, 32, 37, 41, 50, 57, 46, 71, 69, 85], // 5
			[16, 13, 7, 37, 41, 27, 57, 46, 45, 69, 85, 75], // 6
			[13, 7, 4, 41, 27, 40, 46, 45, 55, 85, 75, 80], // 7
			[7, 4, 20, 27, 40, 28, 45, 55, 62, 75, 80, 79], // 8
			[4, 20, 18, 40, 28, 24, 55, 62, 58, 80, 79, 66], // 9
			[20, 18, 14, 28, 24, 36, 62, 58, 47, 79, 66, 83], // 10
			[18, 14, 0, 24, 36, 30, 58, 47, 52, 66, 83, 82], // 11
			[14, 0, 1, 36, 30, 26, 47, 52, 51, 83, 82, 78], // 12
			[0, 1, 3, 30, 26, 25, 52, 51, 63, 82, 78, 87], // 13
			[1, 3, 17, 26, 25, 42, 51, 63, 48, 78, 87, 86], // 14
			[3, 17, 12, 25, 42, 43, 63, 48, 60, 87, 86, 72], // 15
			[17, 12, 8, 42, 43, 35, 48, 60, 49, 86, 72, 81], // 16
			[12, 8, 19, 43, 35, 22, 60, 49, 64, 72, 81, 70], // 17
			[8, 19, 9, 35, 22, 23, 49, 64, 65, 81, 70, 84], // 18
			[19, 9, 11, 22, 23, 38, 64, 65, 56, 70, 84, 74], // 19
			[9, 11, 21, 23, 38, 31, 65, 56, 54, 84, 74, 67], // 20
			[11, 21, 5, 38, 31, 39, 56, 54, 53, 74, 67, 68], // 21
			[10, 3, 2, 34, 30, 23, 63, 57, 59, 85, 86, 81], // 22
			[3, 2, 7, 30, 23, 38, 57, 59, 47, 86, 81, 76], // 23
			[2, 7, 12, 23, 38, 32, 59, 47, 51, 81, 76, 71], // 24
			[7, 12, 16, 38, 32, 26, 47, 51, 46, 76, 71, 77], // 25
			[12, 16, 18, 32, 26, 24, 51, 46, 44, 71, 77, 87], // 26
			[16, 18, 14, 26, 24, 40, 46, 44, 60, 77, 87, 82], // 27
			[18, 14, 11, 24, 40, 39, 44, 60, 58, 87, 82, 74], // 28
			[14, 11, 0, 40, 39, 42, 60, 58, 61, 82, 74, 68], // 29
			[11, 0, 9, 39, 42, 31, 58, 61, 45, 74, 68, 83], // 30
			[0, 9, 15, 42, 31, 22, 61, 45, 54, 68, 83, 70], // 31
			[9, 15, 4, 31, 22, 28, 45, 54, 56, 83, 70, 73], // 32
			[15, 4, 1, 22, 28, 35, 54, 56, 62, 70, 73, 75], // 33
			[4, 1, 8, 28, 35, 41, 56, 62, 55, 73, 75, 67], // 34
			[1, 8, 17, 35, 41, 36, 62, 55, 50, 75, 67, 66], // 35
			[8, 17, 19, 41, 36, 27, 55, 50, 49, 67, 66, 80], // 36
			[17, 19, 20, 36, 27, 37, 50, 49, 53, 66, 80, 72], // 37
			[19, 20, 21, 27, 37, 43, 49, 53, 52, 80, 72, 84], // 38
			[20, 21, 5, 37, 43, 25, 53, 52, 48, 72, 84, 69], // 39
			[21, 5, 13, 43, 25, 33, 52, 48, 64, 84, 69, 78], // 40
			[5, 13, 6, 25, 33, 29, 48, 64, 65, 69, 78, 79], // 41
			[13, 6, 10, 33, 29, 34, 64, 65, 63, 78, 79, 85], // 42
			[6, 10, 3, 29, 34, 30, 65, 63, 57, 79, 85, 86], // 43
			[13, 9, 7, 41, 39, 34, 64, 51, 59, 67, 84, 77], // 44
			[9, 7, 5, 39, 34, 40, 51, 59, 47, 84, 77, 72], // 45
			[7, 5, 19, 34, 40, 42, 59, 47, 58, 77, 72, 82], // 46
			[5, 19, 10, 40, 42, 37, 47, 58, 52, 72, 82, 81], // 47
			[19, 10, 17, 42, 37, 35, 58, 52, 55, 82, 81, 73], // 48
			[10, 17, 18, 37, 35, 28, 52, 55, 65, 81, 73, 70], // 49
			[17, 18, 3, 35, 28, 25, 55, 65, 60, 73, 70, 78], // 50
			[18, 3, 16, 28, 25, 43, 65, 60, 44, 70, 78, 76], // 51
			[3, 16, 20, 25, 43, 36, 60, 44, 61, 78, 76, 75], // 52
			[16, 20, 21, 43, 36, 27, 44, 61, 53, 76, 75, 71], // 53
			[20, 21, 4, 36, 27, 31, 61, 53, 62, 75, 71, 79], // 54
			[21, 4, 11, 27, 31, 33, 53, 62, 46, 71, 79, 80], // 55
			[4, 11, 2, 31, 33, 32, 62, 46, 50, 79, 80, 69], // 56
			[11, 2, 0, 33, 32, 24, 46, 50, 63, 80, 69, 87], // 57
			[2, 0, 8, 32, 24, 23, 50, 63, 54, 69, 87, 66], // 58
			[0, 8, 1, 24, 23, 26, 63, 54, 56, 87, 66, 85], // 59
			[8, 1, 6, 23, 26, 22, 54, 56, 57, 66, 85, 86], // 60
			[1, 6, 15, 26, 22, 38, 56, 57, 48, 85, 86, 68], // 61
			[6, 15, 12, 22, 38, 30, 57, 48, 45, 86, 68, 74], // 62
			[15, 12, 14, 38, 30, 29, 48, 45, 49, 68, 74, 83], // 63
			[12, 14, 13, 30, 29, 41, 45, 49, 64, 74, 83, 67], // 64
			[14, 13, 9, 29, 41, 39, 49, 64, 51, 83, 67, 84], // 65
			[16, 0, 6, 29, 38, 36, 60, 54, 55, 68, 78, 71], // 66
			[0, 6, 4, 38, 36, 39, 54, 55, 45, 78, 71, 70], // 67
			[6, 4, 15, 36, 39, 41, 55, 45, 56, 71, 70, 73], // 68
			[4, 15, 1, 39, 41, 31, 45, 56, 57, 70, 73, 80], // 69
			[15, 1, 13, 41, 31, 26, 56, 57, 61, 73, 80, 66], // 70
			[1, 13, 2, 31, 26, 27, 57, 61, 63, 80, 66, 86], // 71
			[13, 2, 19, 26, 27, 35, 61, 63, 52, 66, 86, 75], // 72
			[2, 19, 8, 27, 35, 30, 63, 52, 51, 86, 75, 72], // 73
			[19, 8, 18, 35, 30, 34, 52, 51, 47, 75, 72, 84], // 74
			[8, 18, 21, 30, 34, 40, 51, 47, 62, 72, 84, 81], // 75
			[18, 21, 7, 34, 40, 25, 47, 62, 65, 84, 81, 67], // 76
			[21, 7, 3, 40, 25, 22, 62, 65, 49, 81, 67, 74], // 77
			[7, 3, 14, 25, 22, 37, 65, 49, 64, 67, 74, 82], // 78
			[3, 14, 10, 22, 37, 33, 49, 64, 46, 74, 82, 87], // 79
			[14, 10, 5, 37, 33, 24, 64, 46, 44, 82, 87, 79], // 80
			[10, 5, 17, 33, 24, 43, 46, 44, 50, 87, 79, 77], // 81
			[5, 17, 9, 24, 43, 42, 44, 50, 59, 79, 77, 83], // 82
			[17, 9, 12, 43, 42, 23, 50, 59, 53, 77, 83, 76], // 83
			[9, 12, 11, 42, 23, 28, 59, 53, 58, 83, 76, 85], // 84
			[12, 11, 20, 23, 28, 32, 53, 58, 48, 76, 85, 69], // 85
			[11, 20, 16, 28, 32, 29, 58, 48, 60, 85, 69, 68], // 86
			[20, 16, 0, 32, 29, 38, 48, 60, 54, 69, 68, 78], // 87
			[8, 5, 16, 43, 37, 23, 55, 56, 49, 87, 70, 66], // 88
			[5, 16, 7, 37, 23, 25, 56, 49, 46, 70, 66, 72], // 89
			[16, 7, 4, 23, 25, 28, 49, 46, 47, 66, 72, 82], // 90
			[7, 4, 12, 25, 28, 41, 46, 47, 45, 72, 82, 71], // 91
			[4, 12, 20, 28, 41, 31, 47, 45, 57, 82, 71, 73], // 92
			[12, 20, 13, 41, 31, 29, 45, 57, 65, 71, 73, 84], // 93
			[20, 13, 19, 31, 29, 35, 57, 65, 60, 73, 84, 69], // 94
			[13, 19, 21, 29, 35, 26, 65, 60, 48, 84, 69, 74], // 95
			[19, 21, 14, 35, 26, 22, 60, 48, 44, 69, 74, 76], // 96
			[21, 14, 1, 26, 22, 27, 48, 44, 51, 74, 76, 78], // 97
			[14, 1, 9, 22, 27, 36, 44, 51, 58, 76, 78, 75], // 98
			[1, 9, 18, 27, 36, 24, 51, 58, 64, 78, 75, 67], // 99
			[9, 18, 0, 36, 24, 42, 58, 64, 50, 75, 67, 80], // 100
			[18, 0, 3, 24, 42, 30, 64, 50, 54, 67, 80, 85], // 101
			[0, 3, 11, 42, 30, 34, 50, 54, 59, 80, 85, 83], // 102
			[3, 11, 2, 30, 34, 40, 54, 59, 52, 85, 83, 86], // 103
			[11, 2, 17, 34, 40, 32, 59, 52, 53, 83, 86, 68], // 104
			[2, 17, 10, 40, 32, 39, 52, 53, 62, 86, 68, 77], // 105
			[17, 10, 15, 32, 39, 38, 53, 62, 61, 68, 77, 79], // 106
			[10, 15, 6, 39, 38, 33, 62, 61, 63, 77, 79, 81], // 107
			[15, 6, 8, 38, 33, 43, 61, 63, 55, 79, 81, 87], // 108
			[6, 8, 5, 33, 43, 37, 63, 55, 56, 81, 87, 70], // 109
			[11, 7, 19, 34, 23, 32, 65, 47, 44, 79, 77, 78], // 110
			[7, 19, 20, 23, 32, 26, 47, 44, 46, 77, 78, 76], // 111
			[19, 20, 1, 32, 26, 28, 44, 46, 58, 78, 76, 82], // 112
			[20, 1, 0, 26, 28, 30, 46, 58, 63, 76, 82, 80], // 113
			[1, 0, 16, 28, 30, 33, 58, 63, 64, 82, 80, 81], // 114
			[0, 16, 4, 30, 33, 24, 63, 64, 53, 80, 81, 68], // 115
			[16, 4, 9, 33, 24, 38, 64, 53, 55, 81, 68, 74], // 116
			[4, 9, 13, 24, 38, 22, 53, 55, 49, 68, 74, 72], // 117
			[9, 13, 14, 38, 22, 37, 55, 49, 50, 74, 72, 70], // 118
			[13, 14, 12, 22, 37, 35, 49, 50, 61, 72, 70, 84], // 119
			[14, 12, 18, 37, 35, 25, 50, 61, 60, 70, 84, 69], // 120
			[12, 18, 8, 35, 25, 42, 61, 60, 48, 84, 69, 67], // 121
			[18, 8, 2, 25, 42, 39, 60, 48, 59, 69, 67, 71], // 122
			[8, 2, 6, 42, 39, 40, 48, 59, 51, 67, 71, 73], // 123
			[2, 6, 21, 39, 40, 41, 59, 51, 54, 71, 73, 83], // 124
			[6, 21, 15, 40, 41, 36, 51, 54, 45, 73, 83, 85], // 125
			[21, 15, 5, 41, 36, 27, 54, 45, 52, 83, 85, 86], // 126
			[15, 5, 17, 36, 27, 31, 45, 52, 57, 85, 86, 66], // 127
			[5, 17, 3, 27, 31, 29, 52, 57, 56, 86, 66, 87], // 128
			[17, 3, 10, 31, 29, 43, 57, 56, 62, 66, 87, 75], // 129
			[3, 10, 11, 29, 43, 34, 56, 62, 65, 87, 75, 79], // 130
			[10, 11, 7, 43, 34, 23, 62, 65, 47, 75, 79, 77] // 131
		];

		var condNum = parseInt(res.condNum);
		if (condNum >= 0 && condNum <= conditions.length - 1) {
			var selectedTrials = conditions[condNum];
		} else {
			var selectedTrials = conditions[defaultCondNum];
			condNum = defaultCondNum * -1;
		}

		maintaskParam = new SetMaintaskParam(selectedTrials);

		// Updates the progress bar
		$("#trial-num").html(maintaskParam.numComplete);
		$("#total-num").html(maintaskParam.numTrials);


		// var maintaskParam = new SetMaintaskParam();
		maintask = {

			stimIDArray: new Array(maintaskParam.numTrials),
			stimulusArray: new Array(maintaskParam.numTrials),

			q1responseArray: new Array(maintaskParam.numTrials),
			q2responseArray: new Array(maintaskParam.numTrials),
			q3responseArray: new Array(maintaskParam.numTrials),
			q4responseArray: new Array(maintaskParam.numTrials),
			q5responseArray: new Array(maintaskParam.numTrials),
			q6responseArray: new Array(maintaskParam.numTrials),
			q7responseArray: new Array(maintaskParam.numTrials),
			q8responseArray: new Array(maintaskParam.numTrials),
			q9responseArray: new Array(maintaskParam.numTrials),
			q10responseArray: new Array(maintaskParam.numTrials),
			q11responseArray: new Array(maintaskParam.numTrials),
			q12responseArray: new Array(maintaskParam.numTrials),
			q13responseArray: new Array(maintaskParam.numTrials),
			q14responseArray: new Array(maintaskParam.numTrials),
			q15responseArray: new Array(maintaskParam.numTrials),
			q16responseArray: new Array(maintaskParam.numTrials),
			q17responseArray: new Array(maintaskParam.numTrials),
			q18responseArray: new Array(maintaskParam.numTrials),
			q19responseArray: new Array(maintaskParam.numTrials),
			q20responseArray: new Array(maintaskParam.numTrials),

			randCondNum: new Array(1),

			validationRadioExpectedResp: new Array(2),
			validationRadio: new Array(3),
			dem_gender: [],
			dem_language: [],
			val_recognized: [],
			val_feedback: [],

			data: [],
			dataInSitu: [],

			validate0: function(expectedResponse) {
				
				var radios = document.getElementsByName('v0');
				var radiosValue = false;

				for (var i = 0; i < radios.length; i++) {
					if (radios[i].checked == true) {
						radiosValue = true;
					}
				}
				if (!radiosValue) {
					alert("Please watch the video and answer the question");
					return false;
				} else {
					this.validationRadioExpectedResp[0] = expectedResponse;
					this.validationRadio[0] = $('input[name="v0"]:checked').val();
					return true;
				}
			},

			validate1: function(expectedResponse) {
				this.validationRadioExpectedResp[1] = expectedResponse;
				this.validationRadio[1] = $('input[name="v1"]:checked').val();
				showSlide('validation2');
			},

			validate2: function(expectedResponse) {
				this.validationRadioExpectedResp[2] = expectedResponse;
				this.validationRadio[2] = $('input[name="v2"]:checked').val();
				showSlide('final');
			},

			end: function() {
				// if (!!subjectValid) {
				// SEND DATA TO TURK
				// }

				this.dem_gender = $('input[name="d1"]:checked').val();;
				this.dem_language = $('textarea[name="dem_language"]').val();
				this.val_recognized = $('textarea[name="val_recognized"]').val();
				this.val_feedback = $('textarea[name="val_feedback"]').val();

				// SEND DATA TO TURK
				setTimeout(function() {
					turk.submit(maintask, true, mmtofurkeyGravy);
					setTimeout(function() { showSlide("exit"); }, 1000);
				}, 1000);

				// DEBUG PUT THIS IN MMTOFURKYGRAVY AND ADD STRING PARAM TO MAINTASK VARIABLES
				console.log("attempting to return condition");
				var returnServe = 'nReps=' + phpParam.nReps + '&writestatus=' + phpParam.writestatus + '&condComplete=' + maintask.randCondNum.toString() + '&condfname=' + phpParam.condfname;
				$.getJSON(phpParam.baseURL, returnServe, function(res) {
					console.log("Serve Returned!", res.condNum);
				});

				// Show the finish slide.
				$('#pietimerelement').pietimer({
					seconds: 2,
					color: 'rgb(76, 76, 76)',
					height: 200,
					width: 200
				});
				showSlide("finished");
				$('#pietimerelement').pietimer('start');
			},

			next: function() {
				// Show the experiment slide.
				// $("#videoStimPackageDiv").hide();
				// $("#playButtonContainer").show();
				showSlide("slideStimulusContext");
				// try {
				// var url = URL.revokeObjectURL(document.getElementById("videoStim_small").src); // IE10+
				// } catch (err) {}
				$('#interactionMask').hide();

				// duplicate allTrialOrders
				if (maintaskParam.numComplete === 0) {
					// this.randCondNum.push(condNum); // push randomization number
					this.randCondNum = condNum;
					// disablePlayButton();
					// preloadStim(0); // load first video

					if (!!maintaskParam.storeDataInSitu) {
						for (var i = 0; i < maintaskParam.allTrialOrders.length; i++) {
							var temp = maintaskParam.allTrialOrders[i];
							temp.q1responseArray = "";
							temp.q2responseArray = "";
							temp.q3responseArray = "";
							temp.q4responseArray = "";
							temp.q5responseArray = "";
							temp.q6responseArray = "";
							temp.q7responseArray = "";
							temp.q8responseArray = "";
							temp.q9responseArray = "";
							temp.q10responseArray = "";
							temp.q11responseArray = "";
							temp.q12responseArray = "";
							temp.q13responseArray = "";
							temp.q14responseArray = "";
							temp.q15responseArray = "";
							temp.q16responseArray = "";
							temp.q17responseArray = "";
							temp.q18responseArray = "";
							temp.q19responseArray = "";
							temp.q20responseArray = "";

							this.dataInSitu.push(temp);
							//// CLEAN this up a little?
						}
					}
				}

				// If this is not the first trial, record variables
				if (maintaskParam.numComplete > 0) {

					this.q1responseArray[maintaskParam.numComplete - 1] = e1.value;
					this.q2responseArray[maintaskParam.numComplete - 1] = e2.value;
					this.q3responseArray[maintaskParam.numComplete - 1] = e3.value;
					this.q4responseArray[maintaskParam.numComplete - 1] = e4.value;
					this.q5responseArray[maintaskParam.numComplete - 1] = e5.value;
					this.q6responseArray[maintaskParam.numComplete - 1] = e6.value;
					this.q7responseArray[maintaskParam.numComplete - 1] = e7.value;
					this.q8responseArray[maintaskParam.numComplete - 1] = e8.value;
					this.q9responseArray[maintaskParam.numComplete - 1] = e9.value;
					this.q10responseArray[maintaskParam.numComplete - 1] = e10.value;
					this.q11responseArray[maintaskParam.numComplete - 1] = e11.value;
					this.q12responseArray[maintaskParam.numComplete - 1] = e12.value;
					this.q13responseArray[maintaskParam.numComplete - 1] = e13.value;
					this.q14responseArray[maintaskParam.numComplete - 1] = e14.value;
					this.q15responseArray[maintaskParam.numComplete - 1] = e15.value;
					this.q16responseArray[maintaskParam.numComplete - 1] = e16.value;
					this.q17responseArray[maintaskParam.numComplete - 1] = e17.value;
					this.q18responseArray[maintaskParam.numComplete - 1] = e18.value;
					this.q19responseArray[maintaskParam.numComplete - 1] = e19.value;
					this.q20responseArray[maintaskParam.numComplete - 1] = e20.value;

					maintaskParam.trial.q1responseArray = e1.value;
					maintaskParam.trial.q2responseArray = e2.value;
					maintaskParam.trial.q3responseArray = e3.value;
					maintaskParam.trial.q4responseArray = e4.value;
					maintaskParam.trial.q5responseArray = e5.value;
					maintaskParam.trial.q6responseArray = e6.value;
					maintaskParam.trial.q7responseArray = e7.value;
					maintaskParam.trial.q8responseArray = e8.value;
					maintaskParam.trial.q9responseArray = e9.value;
					maintaskParam.trial.q10responseArray = e10.value;
					maintaskParam.trial.q11responseArray = e11.value;
					maintaskParam.trial.q12responseArray = e12.value;
					maintaskParam.trial.q13responseArray = e13.value;
					maintaskParam.trial.q14responseArray = e14.value;
					maintaskParam.trial.q15responseArray = e15.value;
					maintaskParam.trial.q16responseArray = e16.value;
					maintaskParam.trial.q17responseArray = e17.value;
					maintaskParam.trial.q18responseArray = e18.value;
					maintaskParam.trial.q19responseArray = e19.value;
					maintaskParam.trial.q20responseArray = e20.value;

					if (!!maintaskParam.storeDataInSitu) {
						maintaskParam.trialInSitu.q1responseArray = e1.value;
						maintaskParam.trialInSitu.q2responseArray = e2.value;
						maintaskParam.trialInSitu.q3responseArray = e3.value;
						maintaskParam.trialInSitu.q4responseArray = e4.value;
						maintaskParam.trialInSitu.q5responseArray = e5.value;
						maintaskParam.trialInSitu.q6responseArray = e6.value;
						maintaskParam.trialInSitu.q7responseArray = e7.value;
						maintaskParam.trialInSitu.q8responseArray = e8.value;
						maintaskParam.trialInSitu.q9responseArray = e9.value;
						maintaskParam.trialInSitu.q10responseArray = e10.value;
						maintaskParam.trialInSitu.q11responseArray = e11.value;
						maintaskParam.trialInSitu.q12responseArray = e12.value;
						maintaskParam.trialInSitu.q13responseArray = e13.value;
						maintaskParam.trialInSitu.q14responseArray = e14.value;
						maintaskParam.trialInSitu.q15responseArray = e15.value;
						maintaskParam.trialInSitu.q16responseArray = e16.value;
						maintaskParam.trialInSitu.q17responseArray = e17.value;
						maintaskParam.trialInSitu.q18responseArray = e18.value;
						maintaskParam.trialInSitu.q19responseArray = e19.value;
						maintaskParam.trialInSitu.q20responseArray = e20.value;
					}

					this.data.push(maintaskParam.trial);

					ResetRanges();
				}

				// If subject has completed all trials, update progress bar and
				// show slide to ask for demographic info
				if (maintaskParam.numComplete >= maintaskParam.numTrials) {
					showSlide("validation1");
					// Update progress bar
					$('.bar').css('width', Math.round(300.0 * maintaskParam.numComplete / maintaskParam.numTrials) + 'px');
					$("#trial-num").html(maintaskParam.numComplete);
					$("#total-num").html(maintaskParam.numTrials);

					// Otherwise, if trials not completed yet, update progress bar
					// and go to next trial based on the order in which trials are supposed
					// to occur
				} else {
					//currentTrialNum is used for randomizing later
					maintaskParam.currentTrialNum = maintaskParam.shuffledOrder[maintaskParam.numComplete]; //numComplete //allTrialOrders[numComplete];
					maintaskParam.trial = maintaskParam.allTrialOrders[maintaskParam.currentTrialNum];
					if (!!maintaskParam.storeDataInSitu) {
						maintaskParam.trialInSitu = this.dataInSitu[maintaskParam.currentTrialNum];
					}

					/// document.getElementById("videoStim").src = serverRoot + "dynamics/" + maintaskParam.allTrialOrders[maintaskParam.shuffledOrder[maintaskParam.numComplete]].stimulus + "t.mp4";
					/// enablePlayButton();

					document.getElementById("imageStim").src = serverRoot + "statics/" + maintaskParam.trial.stimulus + ".png";

					if (maintaskParam.trial.stimulus[4] == 1) {
						document.getElementById("imageStim_front1").src = document.getElementById("imageStim").src;
						document.getElementById("imageStim_front2").src = serverRoot1 + "images/generic_avatar_male.png";
						maintaskParam.trial.decisionOther
					} else if (maintaskParam.trial.stimulus[4] == 2) {
						document.getElementById("imageStim_front1").src = serverRoot1 + "images/generic_avatar_male.png";
						document.getElementById("imageStim_front2").src = document.getElementById("imageStim").src;
					}


					/// console.log("currentImg", maintaskParam.trial.stimulus);
					$('#context_jackpot').text("$" + numberWithCommas(maintaskParam.trial.pot));
					$('#context_jackpot_front').text("$" + numberWithCommas(maintaskParam.trial.pot));

					$('#contextText_decisionOther').html("&nbsp;" + maintaskParam.trial.decisionOther);
					$('#contextText_decisionThis').html("&nbsp;" + maintaskParam.trial.decisionThis);
					document.getElementById("contextImg_decisionOther").src = serverRoot1 + "images/" + maintaskParam.trial.decisionOther + "Ball.png";
					document.getElementById("contextImg_decisionThis").src = serverRoot1 + "images/" + maintaskParam.trial.decisionThis + "Ball.png";
					document.getElementById("miniface_Other").src = serverRoot1 + "images/generic_avatar_male.png";
					document.getElementById("miniface_This").src = document.getElementById("imageStim").src;

					var outcomeOther = 0;
					var outcomeThis = 0;
					if (maintaskParam.trial.decisionOther === "Split" && maintaskParam.trial.decisionThis === "Split") {
						outcomeOther = 'Won $' + numberWithCommas(Math.floor(maintaskParam.trial.pot * 50) / 100);
						outcomeThis = outcomeOther;

						document.getElementById("imageContext").src = serverRoot1 + "images/" + "CC.png";
						$('#context_outcome_front1').html(outcomeOther);
						$('#context_outcome_front2').html(outcomeOther);

					}
					if (maintaskParam.trial.decisionOther === "Split" && maintaskParam.trial.decisionThis === "Stole") {
						outcomeOther = 'Won $0.00';
						outcomeThis = 'Won $' + numberWithCommas(maintaskParam.trial.pot);

						if (maintaskParam.trial.stimulus[4] == 1) {
							document.getElementById("imageContext").src = serverRoot1 + "images/" + "DC.png";
							$('#context_outcome_front1').html(outcomeThis);
							$('#context_outcome_front2').html(outcomeOther);
						} else if (maintaskParam.trial.stimulus[4] == 2) {
							document.getElementById("imageContext").src = serverRoot1 + "images/" + "CD.png";
							$('#context_outcome_front1').html(outcomeOther);
							$('#context_outcome_front2').html(outcomeThis);
						}


					}
					if (maintaskParam.trial.decisionOther === "Stole" && maintaskParam.trial.decisionThis === "Split") {
						outcomeOther = 'Won $' + numberWithCommas(maintaskParam.trial.pot);
						outcomeThis = 'Won $0.00';

						if (maintaskParam.trial.stimulus[4] == 1) {
							document.getElementById("imageContext").src = serverRoot1 + "images/" + "CD.png";
							$('#context_outcome_front1').html(outcomeThis);
							$('#context_outcome_front2').html(outcomeOther);
						} else if (maintaskParam.trial.stimulus[4] == 2) {
							document.getElementById("imageContext").src = serverRoot1 + "images/" + "DC.png";
							$('#context_outcome_front1').html(outcomeOther);
							$('#context_outcome_front2').html(outcomeThis);
						}

					}
					if (maintaskParam.trial.decisionOther === "Stole" && maintaskParam.trial.decisionThis === "Stole") {
						outcomeOther = 'Won $0.00';
						outcomeThis = 'Won $0.00';

						document.getElementById("imageContext").src = serverRoot1 + "images/" + "DD.png";
						$('#context_outcome_front1').html(outcomeOther);
						$('#context_outcome_front2').html(outcomeOther);
					}
					$('#context_outcomeOther').html(outcomeOther);
					$('#context_outcomeThis').html(outcomeThis);

					$("#contextTableFrontDiv").html('&nbsp;');
					$("#contextSubTableID").clone().appendTo("#contextTableFrontDiv"); // insert information in video div

					this.stimIDArray[maintaskParam.numComplete] = maintaskParam.trial.stimID;
					this.stimulusArray[maintaskParam.numComplete] = maintaskParam.trial.stimulus;

					maintaskParam.numComplete++;

					// Update progress bar
					$('.bar').css('width', Math.round(300.0 * maintaskParam.numComplete / maintaskParam.numTrials) + 'px');
					$("#trial-num").html(maintaskParam.numComplete);
					$("#total-num").html(maintaskParam.numTrials);
				}
				// var startTime = (new Date()).getTime();
				// var endTime = (new Date()).getTime();
				//key = (keyCode == 80) ? "p" : "q",
				//userParity = experiment.keyBindings[key],
				// data = {
				//   stimulus: n,
				//   accuracy: realParity == userParity ? 1 : 0,
				//   rt: endTime - startTime
				// };

				// experiment.data.push(data);
				//setTimeout(experiment.next, 500);
			}
		};

	});
}
