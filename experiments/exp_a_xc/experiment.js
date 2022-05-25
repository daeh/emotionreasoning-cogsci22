///////////// head /////////////

var debug = false,
	revoke = false;

/// initalize sliders ///
$('.not-clicked').mousedown(function() {
	$(this).removeClass('not-clicked').addClass('slider-input');
	$(this).closest('.slider').children('.slider-label-not-clicked').removeClass('slider-label-not-clicked').addClass('slider-label');
});

/// initalize rollover effects of emotion labels ///
$('.confidenceRatingRow').hover(
	function() {
		$(this).closest("tr").children("td").children("span.eFloor").html('split&nbsp;');
		$(this).closest("tr").children("td").children("span.eFloor").closest("td").mousedown(function() {
			$(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val('0');
			$(this).closest("tr").children("td").children("div.slider").children('.slider-label-not-clicked').removeClass('slider-label-not-clicked').addClass('slider-label');
			$(this).closest("tr").children("td").children("div.slider").children('.not-clicked').removeClass('not-clicked').addClass('slider-input');
			$(this).closest("tr").children("td").children("span.eFloorPercent").html(100-$(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val() + '&nbsp;');
			$(this).closest("tr").children("td").children("span.eCeilingPercent").html('&nbsp;' + $(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val());
			// $("#barChart").height(Math.round((getResponseDivSizes() / 48) * $(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val()));
		});
		$(this).closest("tr").children("td").children("span.eCeiling").html('&nbsp;steal');
		$(this).closest("tr").children("td").children("span.eCeiling").closest("td").mousedown(function() {
			$(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val('100');
			$(this).closest("tr").children("td").children("div.slider").children('.slider-label-not-clicked').removeClass('slider-label-not-clicked').addClass('slider-label');
			$(this).closest("tr").children("td").children("div.slider").children('.not-clicked').removeClass('not-clicked').addClass('slider-input');
			$(this).closest("tr").children("td").children("span.eFloorPercent").html(100-$(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val() + '&nbsp;');
			$(this).closest("tr").children("td").children("span.eCeilingPercent").html('&nbsp;' + $(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val());
			// $("#barChart").height(Math.round((getResponseDivSizes() / 48) * $(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val()));
		});
		// $(this).closest("tr").children("td").children("span.eFloorPercent").html(100-$(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val() + '&nbsp;');
		// $(this).closest("tr").children("td").children("span.eCeilingPercent").html('&nbsp;' + $(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val());
		// $("#barChart").height(Math.round((getResponseDivSizes() / 48) * $(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val()));
	},
	function() {
		// $('#respRightEDisplayText').html('&nbsp;');
		$(this).closest("tr").children("td").children("span.eFloor").html('&nbsp;');
		$(this).closest("tr").children("td").children("span.eCeiling").html('&nbsp;');
		// $("#barChart").height(0);
	}
);

// $('.emotionRatingRow').mouseup(function() {$('#TEMP').html( $(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val() ) } );
// $('.emotionRatingRow').mousemove(function() { $('#TEMP').html($(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val()) });
$('.confidenceRatingRow').click(function() {
	$(this).closest("tr").children("td").children("span.eFloorPercent").html(100-$(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val() + '&nbsp;');
	$(this).closest("tr").children("td").children("span.eCeilingPercent").html('&nbsp;' + $(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val());
	// $(this).closest("tr").children("td").children("textarea.eFloorPercentField").val(100-$(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val() + '&nbsp;');
	// document.getElementById("textField1").value = "abc";
	// $("#barChart").height(
	// 	Math.round((getResponseDivSizes() / 48) * $(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val())
	// );
});
$('.confidenceRatingRow').mousemove(function() {
	$(this).closest("tr").children("td").children("span.eFloorPercent").html(100-$(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val() + '&nbsp;');
	$(this).closest("tr").children("td").children("span.eCeilingPercent").html('&nbsp;' + $(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val());
	// $("#barChart").height(
	// 	Math.round((getResponseDivSizes() / 48) * $(this).closest("tr").children("td").children("div.slider").children('input[type="range"]').val())
	// );
});
// trying to get bar to update on mousedown
// $("div.slider").mousedown(function() {
// 	var aa = $(this).children('input[type="range"]').val();
// console.log('aa: ',aa);
// });


/// initalize opacity changes for ball choices (exp5) ///
BTS_actual_otherPlayDecisionOptions.addEventListener("click", adjustConfidence, false);
BTS_actual_thisPlayDecisionOptions.addEventListener("click", adjustConfidence, false);
// BTS_predicted_otherPlayDecisionOptions.addEventListener("click", adjustConfidence, false);
// BTS_predicted_thisPlayDecisionOptions.addEventListener("click", adjustConfidence, false);

BTS_toggle.addEventListener("click", toggleBTSpane, false);

var confidenceOpacityMap = [[1,0],[0.6,0.05],[0.4,0.1],[0.1,0.4],[0.05,0.6],[0,1]];
var payoffMatrix = [[[-1, -1, -1, -1],[-1, -1, -1, -1]],[[-1, -1, -1, -1],[-1, -1, -1, -1]]];

function toggleBTSpane(a) {
	if (!document.getElementById("BTS_toggle").checked) {
		// BTS_actual_title.style.display = 'block';
		BTS_actual_thisPlayDecisionOptions.style.display = 'block';
		BTS_actual_otherPlayDecisionOptions.style.display = 'block';

		// BTS_predicted_title.style.display = 'none';
		BTS_predicted_thisPlayDecisionOptions.style.display = 'none';
		BTS_predicted_otherPlayDecisionOptions.style.display = 'none';

		BTS_actual_instructions.style.display = 'block';
		BTS_predicted_instructions.style.display = 'none';
		/*
		var on = document.getElementsByClassName("BTS-actual");
		var off = document.getElementsByClassName("BTS-predicted");

		for (var i = 0; i < on.length; i++) {
			on[i].style.display = 'block';
		}
		*/

		adjustConfidence();
	} else {
		// BTS_actual_title.style.display = 'none';
		BTS_actual_thisPlayDecisionOptions.style.display = 'none';
		BTS_actual_otherPlayDecisionOptions.style.display = 'none';

		// BTS_predicted_title.style.display = 'block';
		BTS_predicted_thisPlayDecisionOptions.style.display = 'block';
		BTS_predicted_otherPlayDecisionOptions.style.display = 'block';

		BTS_actual_instructions.style.display = 'none';
		BTS_predicted_instructions.style.display = 'block';

		document.getElementById("choiceBallMain-left-other").style.opacity = 1;
		document.getElementById("choiceBallMain-right-other").style.opacity = 1;
		document.getElementById("choiceBallMain-left-this").style.opacity = 1;
		document.getElementById("choiceBallMain-right-this").style.opacity = 1;
	}
	// adjustConfidence();
}

function adjustConfidence(a) {
	adjustConfidenceOther();
	adjustConfidenceThis();
	setPayoffMatrix();
}

function adjustConfidenceOther(a) {
	// var opac = Math.round(100 * confidenceThis.value / 48) / 100;
	// if (a.srcElement.tagName === "LABEL") {
	// 	return;
	// }
	var radioValue = NaN;
	var opacLeft = 1.0,
		opacRight = 1.0;
	if (!document.getElementById("BTS_toggle").checked) {
		radioValue = getRadioResponse('BTS_actual-otherPlayer-confidence');
	}
	else {
		radioValue = getRadioResponse('BTS_predicted-otherPlayer-confidence');
	}
	if (!isNaN(radioValue) && radioValue != "") {
		opacLeft = confidenceOpacityMap[radioValue][0];
		opacRight = confidenceOpacityMap[radioValue][1];
	}

	document.getElementById("choiceBallMain-left-other").style.opacity = opacLeft;
	document.getElementById("choiceBallMain-right-other").style.opacity = opacRight;
	document.getElementById("choiceBallMain-left-other").style.filter = 'alpha(opacity=' + Math.round(opacLeft * 100) + ')'; // IE fallback
	document.getElementById("choiceBallMain-right-other").style.filter = 'alpha(opacity=' + Math.round(opacRight * 100) + ')'; // IE fallback
}

function adjustConfidenceThis(a) {
	// var opac = Math.round(100 * confidenceThis.value / 48) / 100;
	// if (a.srcElement.tagName === "LABEL") {
	// 	return;
	// }
	var radioValue = NaN;
	var opacLeft = 1.0,
		opacRight = 1.0;
	if (!document.getElementById("BTS_toggle").checked) {
		radioValue = getRadioResponse('BTS_actual-thisPlayer-confidence');
	}
	else {
		radioValue = getRadioResponse('BTS_predicted-thisPlayer-confidence');
	}
	if (!isNaN(radioValue) && radioValue != "") {
		// debugger;
		opacLeft = confidenceOpacityMap[radioValue][0];
		opacRight = confidenceOpacityMap[radioValue][1];
	}

	document.getElementById("choiceBallMain-left-this").style.opacity = opacLeft;
	document.getElementById("choiceBallMain-right-this").style.opacity = opacRight;
	document.getElementById("choiceBallMain-left-this").style.filter = 'alpha(opacity=' + Math.round(opacLeft * 100) + ')'; // IE fallback
	document.getElementById("choiceBallMain-right-this").style.filter = 'alpha(opacity=' + Math.round(opacRight * 100) + ')'; // IE fallback
}

function setPayoffMatrix() {

	var BTS_actual_otherDecisionConfidence = getRadioResponse('BTS_actual-otherPlayer-confidence');
	var BTS_actual_thisDecisionConfidence = getRadioResponse('BTS_actual-thisPlayer-confidence');

	var BTS_predicted_otherDecisionConfidence = getRadioResponse('BTS_predicted-otherPlayer-confidence');
	var BTS_predicted_thisDecisionConfidence = getRadioResponse('BTS_predicted-thisPlayer-confidence');
	// BTS actual
	// THIS
	if (BTS_actual_thisDecisionConfidence != "") {
		if (BTS_actual_thisDecisionConfidence < 2.5) {
			payoffMatrix[0][0] = [1, 0, 1, 0];
		} else {
			payoffMatrix[0][0] = [0, 1, 0, 1];
		}
	} else {
		payoffMatrix[0][0] = [1, 1, 1, 1];
	}
	
	// OTHER
	if (BTS_actual_otherDecisionConfidence != "") {
		if (BTS_actual_otherDecisionConfidence < 2.5 && BTS_actual_otherDecisionConfidence != "") {
			payoffMatrix[0][1] = [1, 1, 0, 0];
		} else {
			payoffMatrix[0][1] = [0, 0, 1, 1];
		}
	} else {
		payoffMatrix[0][1] = [1, 1, 1, 1];
	}

	// BTS actual
	// THIS
	if (BTS_predicted_thisDecisionConfidence != "") {
		if (BTS_predicted_thisDecisionConfidence < 2.5 && BTS_predicted_thisDecisionConfidence != "") {
			payoffMatrix[1][0] = [1, 0, 1, 0];
		} else {
			payoffMatrix[1][0] = [0, 1, 0, 1];
		}
	} else {
		payoffMatrix[1][0] = [1, 1, 1, 1];
	}

	// OTHER
	if (BTS_predicted_otherDecisionConfidence != "") {
		if (BTS_predicted_otherDecisionConfidence < 2.5 && BTS_predicted_otherDecisionConfidence != "") {
			payoffMatrix[1][1] = [1, 1, 0, 0];
		} else {
			payoffMatrix[1][1] = [0, 0, 1, 1];
		}
	} else {
		payoffMatrix[1][1] = [1, 1, 1, 1];
	}

	updatePayoffMatrix('current');
}

function sumArray(array) {
	var sum = 0;
	for (var i = 0; i < array.length; i++) {
		sum += array[i];
	}
	return sum;
}

function findFirstOne(element) {
  return element === 1;
} 

function updatePayoffMatrix(behavior) {
	var payoffMatrixImages = ['payoffCC','payoffDC','payoffCD','payoffDD'];
	var BTS_index = -1;
	var quadrant = 0;
	if (behavior == 'current') {
		if (!document.getElementById("BTS_toggle").checked) {
			BTS_index = 0;
		} else {
			BTS_index = 1;
		}
	} else if (typeof behavior == 'number') {
		BTS_index = behavior;
	}

	var displayMatrix = [0, 0, 0, 0];
	for (var i = 0; i < payoffMatrix[0][0].length; i++) {
		if (Math.abs(payoffMatrix[BTS_index][0][i] * payoffMatrix[BTS_index][1][i]) > 0) 
		{
			document.getElementById(payoffMatrixImages[i]).style.display = 'block';
			displayMatrix[i] = 1;
		} 
		else {document.getElementById(payoffMatrixImages[i]).style.display = 'none';}

		if (i == payoffMatrix[0][0].length - 1 && sumArray(displayMatrix) == 1) {
			quadrant = displayMatrix.findIndex(findFirstOne) + 1;
			switch(quadrant) {
			    case 1:
					//CC
					$("#moneyother").html('$' + numberWithCommas(Math.floor(maintaskParam.trial.pot * 50) / 100));
					$("#moneythis").html('$' + numberWithCommas(Math.floor(maintaskParam.trial.pot * 50) / 100));
					break;
				case 2:
					//DC
					$("#moneyother").html('$0.00');
					$("#moneythis").html('$' + numberWithCommas(maintaskParam.trial.pot));
					break;
				case 3:
					//CD
					$("#moneyother").html('$' + numberWithCommas(maintaskParam.trial.pot));
					$("#moneythis").html('$0.00');
					break;
				case 4:
					//DD
					$("#moneyother").html('$0.00');
					$("#moneythis").html('$0.00');
					break;
			}
		} else {
			$("#moneyother").html('&nbsp;');
			$("#moneythis").html('&nbsp;');
		}
	}
	return quadrant;

	//console.log('displayMatrix',displayMatrix);
	// return displayMatrix;

	// DISPLAY Qx.png image based on displayMatrix.
	
	// FIND SUM OF displayMatrix; IF SUM === 1, then display $s for players
	// IF displayMatrix[0], display $$ for both players, if displayMatrix[1] or displayMatrix[2] === 1, display $$$$ for the correct players (else do nothing).
}
///////////// Window /////////////


/// MTURK Initalization ///

var iSlide = 0;
var iTrial = 0;
var serverRoot = ""; // requires terminal slash
var stimPath = "../stimuli/"; // requires terminal slash

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

function getRadioResponse(radioName) {
	var radios = document.getElementsByName(radioName);
	for (var i = 0; i < radios.length; i++) {
		if (radios[i].checked == true) {
			return radios[i].value;
		}
	}
	return '';
}

function ResetRadios(radioName) {
	var radios = document.getElementsByName(radioName);
	for (var i = 0; i < radios.length; i++) {
		radios[i].checked = false;
	}
}

function ValidateRadios(radioNameList) {
	for (var j = 0; j < radioNameList.length; j++) {
		var pass = false;
		var radios = document.getElementsByName(radioNameList[j])
		for (var i = 0; i < radios.length; i++) {
			if (radios[i].checked === true) {
				pass = true;
				i = radios.length;
			}
		}
		if (pass === false) {
			alert("Please provide an answer to every question.");
			return pass;
		}
	}
	return pass;
}


// Ranges //
function ValidateRanges(className) {
	var pass = true;
	var unanswered = $("." + className).find(".slider-label-not-clicked");
	if (unanswered.length > 0) {
		pass = false;
		alert("Please provide an answer to every question.");
		//alert("Please provide an answer to all emotions. If you think that a person is not experiencing a given emotion, rate that emotion as --not at all-- by moving the sliding marker all the way to the left.");
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
		ranges[i].value = "50";
	}
	ranges.removeClass('slider-input').addClass('not-clicked');
	ranges.closest('.slider').children('label').removeClass("slider-label").addClass("slider-label-not-clicked");

	ranges.closest("tr").children("td").children("span.eFloorPercent").html('50');
	ranges.closest("tr").children("td").children("span.eCeilingPercent").html('50');
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
}

function presentStim() {
	if (nPresentations === 0) {

		responseDivCues1.style.display = 'none';

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

	toggleBTSpane();

	if (!!revoke) {
		thisVideo.src = URL.revokeObjectURL(smallVideo.src); // IE10+
		thisVideo.src = '';
	}

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

var smallVideo = document.getElementById("videoStim_small");

var thisVideo = document.getElementById("videoStim");
var nPresentations = 0;

function stimHandler() {
	$('#vidoStimFrameID').removeClass('videoStimFramingDivClassRunning').addClass('videoStimFramingDivClass');
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
			document.getElementById("imageStim_preload").src = serverRoot + stimPath + "statics/" + maintaskParam.allTrialOrders[maintaskParam.shuffledOrder[maintaskParam.numComplete]].stimulus + ".png"; // preload next static image
			if (!!debug) {
				preloadStim(maintaskParam.numComplete,false); // load next movie direct
			} else {
				preloadStim(maintaskParam.numComplete,true); // load next movie as blob
			}
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
			console.log('running');
			$('#vidoStimFrameID').removeClass('videoStimFramingDivClass').addClass('videoStimFramingDivClassRunning');
			thisVideo.play();
			document.getElementById('videoStim').addEventListener('ended', stimHandler, false);
		}, 1000);
	}, 1000);
}

/*
// FETCH calls not supported by legacy browsers (or currently iOS). Using XML instead for the time being.
// load next video
function preloadStim(stimNum) {
// var tempurl = serverRoot + stimPath + "dynamics/" + maintaskParam.allTrialOrders[maintaskParam.shuffledOrder[ stimNum ]].stimulus + "t.mp4";
fetch(serverRoot + stimPath + "dynamics/" + maintaskParam.allTrialOrders[maintaskParam.shuffledOrder[stimNum]].stimulus + "t.mp4").then(function(response) {
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

function preloadStim(stimNum,genBlob) {
	var req = new XMLHttpRequest();
	var stimURL = serverRoot + stimPath + "dynamics/" + maintaskParam.allTrialOrders[maintaskParam.shuffledOrder[stimNum]].stimulus + "t.mp4";

	if (!genBlob) {
		console.log("BYPASSING BLOB");
		thisVideo.src = stimURL;
		enablePlayButton();
	} else {
		req.open('GET', stimURL, true);
		req.responseType = 'blob';

		req.onload = function() {
			// Onload is triggered even on 404
			// so we need to check the status code
			if (this.status === 200) {
				var videoBlob = this.response;
				// document.getElementById("videoStim").src = URL.createObjectURL(videoBlob); // IE10+
				thisVideo.src = URL.createObjectURL(videoBlob); // IE10+
				enablePlayButton();
				console.log("Current load: stimNum    ", stimNum, "maintaskParam.shuffledOrder[ stimNum ]                  ", maintaskParam.shuffledOrder[stimNum], "stimID", maintaskParam.allTrialOrders[maintaskParam.shuffledOrder[stimNum]].stimulus, "( maintaskParam.numComplete", maintaskParam.numComplete, ")");
			}
		};
		req.onerror = function() {
			console.log("Booo");
		};
		req.send();
	}
}


// Play button control //
function disablePlayButton() {
	$('#playButton').prop('onclick', null).off('click');
	$('#playButton').removeClass('play-button').addClass('play-button-inactive');
	$("#loadingTextLeft").html('VIDEO&nbsp;');
	$("#loadingTextRight").html('&nbsp;LOADING');

	//FUTURE MAYBE
	if (maintaskParam.numComplete > 0) {
		//thisVideo.src = serverRoot + stimPath + "dynamics/" + maintaskParam.trial.stimulus + "t.mp4";
		/// check if video has loaded
		//enablePlayButton();

		//temp_stimURL = serverRoot + stimPath + "dynamics/" + maintaskParam.trial.stimulus + "t.mp4";
		//$("#responseDivReload").html('<a class="whiteText" onClick="thisVideo.src = temp_stimURL; enablePlayButton(); console.log(temp_stimURL);" style="cursor: pointer; cursor: hand;">*click here*</a>')
	}
}

function enablePlayButton() {
	$('#playButton').click(function() {
		presentStim();
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

	phpSave: function(rawData_floorLevel, turk) {
		if (!turk.assignmentId || !turk.turkSubmitTo) {
			$.ajax ({
				url: "http://daeda.scripts.mit.edu/GB/storejson.php",
				type: "POST",
				data: JSON.stringify(rawData_floorLevel),
				dataType: "json",
				contentType: "application/json; charset=utf-8",
				success: function(){
					console.log("happy eating");
				}
			});

			// $.post("http://daeda.scripts.mit.edu/GB/storejson.php", JSON.stringify(rawData_floorLevel), function(result){
			// 	alert(result);
			// });
		} else { console.log("Skipping Gravy phpSave");}

	},

	saveZip: function(rawData_floorLevel, turk) {
		if (!turk.assignmentId || !turk.turkSubmitTo) {

			var zip = new JSZip();
			zip.file("exp5datadump.txt", JSON.stringify(rawData_floorLevel) + "\n\n");

			zip.generateAsync({type:"blob"}).then(function (blob) { 	// 1) generate the zip file
				saveAs(blob, "exp5datadump.zip");                          // 2) trigger the download
			}, function (err) {
				console.log('zipping error: ', err);
			});
		} else {console.log("Skipping Gravy saveZip");}
	},

	sendEmail: function(rawData_floorLevel, turk) {
		if (!turk.assignmentId || !turk.turkSubmitTo) {
			document.location.href = "mailto:daeda@mit.edu?subject=" + encodeURIComponent("Exp5 backup data") + "&body=" + "Thanks for participating! Please attach the exp5datadump.zip file that (should have) just downloaded to this email and send it along!" + escape('\r\n\r\n\r\n') + encodeURIComponent(JSON.stringify(rawData_floorLevel)) + escape('\r\n\r\n') + "THANKS!" + escape('\r\n\r\n');
		} else {console.log("Skipping Gravy sendEmail");}
	}
};

var trainingVideo = {
	preloadStim: function() {
		var req = new XMLHttpRequest();
		var stimURL = serverRoot + stimPath + "dynamics/" + "258_c_ed_vbr2.mp4";
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
			///FUTURE
		};
		req.send();
	}
};

function enableAdvance() {
	$('#training_advance_button').removeClass('advance-button-inactive').addClass('advance-button');
	$('#training_advance_button').removeClass('advance-button-inactive').addClass('advance-button');
	document.getElementById('training_advance_button').onclick = function() {
		if (!!maintask.validate0('7510')) {
			this.blur();
			showSlide(2);
			document.getElementById("videoStim_training").src = URL.revokeObjectURL(document.getElementById("videoStim_training").src);
			document.getElementById("videoStim_training").src = '';
		}

	};
}

/// Experiment ///

function SetMaintaskParam() {
	this.allConditions = [
		[
			{ "condition": 1, "Version": "1a", "stimID": 235.1, "stimulus": "235_1", "pot": 50221.00 }, // 0
			{ "condition": 1, "Version": "1a", "stimID": 235.2, "stimulus": "235_2", "pot": 50221.00 }, // 1
			{ "condition": 1, "Version": "1a", "stimID": 237.1, "stimulus": "237_1", "pot": 65673.50 }, // 2
			{ "condition": 1, "Version": "1a", "stimID": 237.2, "stimulus": "237_2", "pot": 65673.50 }, // 3
			{ "condition": 1, "Version": "1a", "stimID": 239.1, "stimulus": "239_1", "pot": 94819.00 }, // 4
			{ "condition": 1, "Version": "1a", "stimID": 239.2, "stimulus": "239_2", "pot": 94819.00 }, // 5
			{ "condition": 1, "Version": "1a", "stimID": 240.1, "stimulus": "240_1", "pot": 36159.00 }, // 6
			{ "condition": 1, "Version": "1a", "stimID": 240.2, "stimulus": "240_2", "pot": 36159.00 }, // 7
			{ "condition": 1, "Version": "1a", "stimID": 241.1, "stimulus": "241_1", "pot": 80954.50 }, // 8
			{ "condition": 1, "Version": "1a", "stimID": 241.2, "stimulus": "241_2", "pot": 80954.50 }, // 9
			{ "condition": 1, "Version": "1a", "stimID": 242.1, "stimulus": "242_1", "pot": 1283.50 }, // 10
			{ "condition": 1, "Version": "1a", "stimID": 242.2, "stimulus": "242_2", "pot": 1283.50 }, // 11
			{ "condition": 1, "Version": "1a", "stimID": 243.1, "stimulus": "243_1", "pot": 7726.50 }, // 12
			{ "condition": 1, "Version": "1a", "stimID": 243.2, "stimulus": "243_2", "pot": 7726.50 }, // 13
			{ "condition": 1, "Version": "1a", "stimID": 244.1, "stimulus": "244_1", "pot": 139.00 }, // 14
			{ "condition": 1, "Version": "1a", "stimID": 244.2, "stimulus": "244_2", "pot": 139.00 }, // 15
			{ "condition": 1, "Version": "1a", "stimID": 245.1, "stimulus": "245_1", "pot": 5958.00 }, // 16
			{ "condition": 1, "Version": "1a", "stimID": 245.2, "stimulus": "245_2", "pot": 5958.00 }, // 17
			{ "condition": 1, "Version": "1a", "stimID": 247.1, "stimulus": "247_1", "pot": 3.50 }, // 18
			{ "condition": 1, "Version": "1a", "stimID": 247.2, "stimulus": "247_2", "pot": 3.50 }, // 19
			{ "condition": 1, "Version": "1a", "stimID": 248.1, "stimulus": "248_1", "pot": 31403.00 }, // 20
			{ "condition": 1, "Version": "1a", "stimID": 248.2, "stimulus": "248_2", "pot": 31403.00 }, // 21
			{ "condition": 1, "Version": "1a", "stimID": 249.1, "stimulus": "249_1", "pot": 1562.50 }, // 22
			{ "condition": 1, "Version": "1a", "stimID": 249.2, "stimulus": "249_2", "pot": 1562.50 }, // 23
			{ "condition": 1, "Version": "1a", "stimID": 250.1, "stimulus": "250_1", "pot": 30304.50 }, // 24
			{ "condition": 1, "Version": "1a", "stimID": 250.2, "stimulus": "250_2", "pot": 30304.50 }, // 25
			{ "condition": 1, "Version": "1a", "stimID": 251.1, "stimulus": "251_1", "pot": 2835.50 }, // 26
			{ "condition": 1, "Version": "1a", "stimID": 251.2, "stimulus": "251_2", "pot": 2835.50 }, // 27
			{ "condition": 1, "Version": "1a", "stimID": 252.1, "stimulus": "252_1", "pot": 310.00 }, // 28
			{ "condition": 1, "Version": "1a", "stimID": 252.2, "stimulus": "252_2", "pot": 310.00 }, // 29
			{ "condition": 1, "Version": "1a", "stimID": 253.1, "stimulus": "253_1", "pot": 21719.50 }, // 30
			{ "condition": 1, "Version": "1a", "stimID": 253.2, "stimulus": "253_2", "pot": 21719.50 }, // 31
			{ "condition": 1, "Version": "1a", "stimID": 254.1, "stimulus": "254_1", "pot": 130884.00 }, // 32
			{ "condition": 1, "Version": "1a", "stimID": 254.2, "stimulus": "254_2", "pot": 130884.00 }, // 33
			{ "condition": 1, "Version": "1a", "stimID": 256.1, "stimulus": "256_1", "pot": 2983.00 }, // 34
			{ "condition": 1, "Version": "1a", "stimID": 256.2, "stimulus": "256_2", "pot": 2983.00 }, // 35
			{ "condition": 1, "Version": "1a", "stimID": 257.1, "stimulus": "257_1", "pot": 30.00 }, // 36
			{ "condition": 1, "Version": "1a", "stimID": 257.2, "stimulus": "257_2", "pot": 30.00 }, // 37
			{ "condition": 1, "Version": "1a", "stimID": 260.1, "stimulus": "260_1", "pot": 1598.50 }, // 38
			{ "condition": 1, "Version": "1a", "stimID": 260.2, "stimulus": "260_2", "pot": 1598.50 }, // 39
			{ "condition": 1, "Version": "1a", "stimID": 262.1, "stimulus": "262_1", "pot": 56488.00 }, // 40
			{ "condition": 1, "Version": "1a", "stimID": 262.2, "stimulus": "262_2", "pot": 56488.00 }, // 41
			{ "condition": 1, "Version": "1a", "stimID": 263.1, "stimulus": "263_1", "pot": 28802.00 }, // 42
			{ "condition": 1, "Version": "1a", "stimID": 263.2, "stimulus": "263_2", "pot": 28802.00 }, // 43
			{ "condition": 1, "Version": "1a", "stimID": 264.1, "stimulus": "264_1", "pot": 24145.00 }, // 44
			{ "condition": 1, "Version": "1a", "stimID": 264.2, "stimulus": "264_2", "pot": 24145.00 }, // 45
			{ "condition": 1, "Version": "1a", "stimID": 265.1, "stimulus": "265_1", "pot": 1588.00 }, // 46
			{ "condition": 1, "Version": "1a", "stimID": 265.2, "stimulus": "265_2", "pot": 1588.00 }, // 47
			{ "condition": 1, "Version": "1a", "stimID": 266.1, "stimulus": "266_1", "pot": 6559.00 }, // 48
			{ "condition": 1, "Version": "1a", "stimID": 266.2, "stimulus": "266_2", "pot": 6559.00 }, // 49
			{ "condition": 1, "Version": "1a", "stimID": 268.1, "stimulus": "268_1", "pot": 61381.50 }, // 50
			{ "condition": 1, "Version": "1a", "stimID": 268.2, "stimulus": "268_2", "pot": 61381.50 }, // 51
			{ "condition": 1, "Version": "1a", "stimID": 269.1, "stimulus": "269_1", "pot": 2318.00 }, // 52
			{ "condition": 1, "Version": "1a", "stimID": 269.2, "stimulus": "269_2", "pot": 2318.00 }, // 53
			{ "condition": 1, "Version": "1a", "stimID": 270.1, "stimulus": "270_1", "pot": 5322.50 }, // 54
			{ "condition": 1, "Version": "1a", "stimID": 270.2, "stimulus": "270_2", "pot": 5322.50 }, // 55
			{ "condition": 1, "Version": "1a", "stimID": 271.1, "stimulus": "271_1", "pot": 19025.50 }, // 56
			{ "condition": 1, "Version": "1a", "stimID": 271.2, "stimulus": "271_2", "pot": 19025.50 }, // 57
			{ "condition": 1, "Version": "1a", "stimID": 272.1, "stimulus": "272_1", "pot": 81899.00 }, // 58
			{ "condition": 1, "Version": "1a", "stimID": 272.2, "stimulus": "272_2", "pot": 81899.00 }, // 59
			{ "condition": 1, "Version": "1a", "stimID": 273.1, "stimulus": "273_1", "pot": 9675.00 }, // 60
			{ "condition": 1, "Version": "1a", "stimID": 273.2, "stimulus": "273_2", "pot": 9675.00 }, // 61
			{ "condition": 1, "Version": "1a", "stimID": 276.1, "stimulus": "276_1", "pot": 130944.00 }, // 62
			{ "condition": 1, "Version": "1a", "stimID": 276.2, "stimulus": "276_2", "pot": 130944.00 }, // 63
			{ "condition": 1, "Version": "1a", "stimID": 277.1, "stimulus": "277_1", "pot": 3331.00 }, // 64
			{ "condition": 1, "Version": "1a", "stimID": 277.2, "stimulus": "277_2", "pot": 3331.00 }, // 65
			{ "condition": 1, "Version": "1a", "stimID": 278.1, "stimulus": "278_1", "pot": 40606.00 }, // 66
			{ "condition": 1, "Version": "1a", "stimID": 278.2, "stimulus": "278_2", "pot": 40606.00 }, // 67
			{ "condition": 1, "Version": "1a", "stimID": 279.1, "stimulus": "279_1", "pot": 269.50 }, // 68
			{ "condition": 1, "Version": "1a", "stimID": 279.2, "stimulus": "279_2", "pot": 269.50 }, // 69
			{ "condition": 1, "Version": "1a", "stimID": 281.1, "stimulus": "281_1", "pot": 1803.00 }, // 70
			{ "condition": 1, "Version": "1a", "stimID": 281.2, "stimulus": "281_2", "pot": 1803.00 }, // 71
			{ "condition": 1, "Version": "1a", "stimID": 282.1, "stimulus": "282_1", "pot": 2843.50 }, // 72
			{ "condition": 1, "Version": "1a", "stimID": 282.2, "stimulus": "282_2", "pot": 2843.50 }, // 73
			{ "condition": 1, "Version": "1a", "stimID": 283.1, "stimulus": "283_1", "pot": 57518.00 }, // 74
			{ "condition": 1, "Version": "1a", "stimID": 283.2, "stimulus": "283_2", "pot": 57518.00 }, // 75
			{ "condition": 1, "Version": "1a", "stimID": 284.1, "stimulus": "284_1", "pot": 1116.50 }, // 76
			{ "condition": 1, "Version": "1a", "stimID": 284.2, "stimulus": "284_2", "pot": 1116.50 }, // 77
			{ "condition": 1, "Version": "1a", "stimID": 285.1, "stimulus": "285_1", "pot": 2300.50 }, // 78
			{ "condition": 1, "Version": "1a", "stimID": 285.2, "stimulus": "285_2", "pot": 2300.50 }, // 79
			{ "condition": 1, "Version": "1a", "stimID": 286.1, "stimulus": "286_1", "pot": 3532.50 }, // 80
			{ "condition": 1, "Version": "1a", "stimID": 286.2, "stimulus": "286_2", "pot": 3532.50 }, // 81
			{ "condition": 1, "Version": "1a", "stimID": 287.1, "stimulus": "287_1", "pot": 1030.50 }, // 82
			{ "condition": 1, "Version": "1a", "stimID": 287.2, "stimulus": "287_2", "pot": 1030.50 }, // 83
			{ "condition": 1, "Version": "1a", "stimID": 288.1, "stimulus": "288_1", "pot": 15744.50 }, // 84
			{ "condition": 1, "Version": "1a", "stimID": 288.2, "stimulus": "288_2", "pot": 15744.50 }, // 85
			{ "condition": 1, "Version": "1a", "stimID": 289.1, "stimulus": "289_1", "pot": 822.50 }, // 86
			{ "condition": 1, "Version": "1a", "stimID": 289.2, "stimulus": "289_2", "pot": 822.50 } // 87
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
	if (!!debug) {
		this.shuffledOrder = shuffleArray(genIntRange(0, 2)); //DEBUG
	// this.shuffledOrder = shuffleArray(selectedTrials);
	} else {
		this.shuffledOrder = shuffleArray(genIntRange(0, this.allTrialOrders.length - 1)); // For all trials
	}
	
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

	this.storeDataInSitu = true;
}

var phpParam = { baseURL: 'https://daeda.scripts.mit.edu/GB/serveCondition.php?callback=?', nReps: 1, condfname: "servedConditions.csv" };
var maintask = [];
var maintaskParam = [];

function loadHIT(nextSlide) {
	// determine if HIT is live
	var assignmentId_local = turk.assignmentId,
		turkSubmitTo_local = turk.turkSubmitTo;

	// If there's no turk info
	// phpParam.writestatus = "TRUE";
	if (!assignmentId_local || !turkSubmitTo_local) {
		console.log("Dead Turkey: Not writing to conditions file.");
		// phpParam.writestatus = "FALSE";
	} else {
		console.log("Live Turkey!");
		// phpParam.writestatus = "TRUE";
	}

	// var requestService = 'nReps=' + phpParam.nReps + '&writestatus=' + phpParam.writestatus + '&condComplete=' + 'REQUEST' + '&condfname=' + phpParam.condfname;

	showSlide("loadingHIT");
	// $.getJSON(phpParam.baseURL, requestService, function(res) {

	// console.log("Served Condition:", res.condNum);

	showSlide(nextSlide);
	var defaultCondNum = 0; // if PHP runs out of options
	var conditions = [
		[0]
	];

	/*
	var condNum = parseInt(res.condNum);
	if (condNum >= 0 && condNum <= conditions.length - 1) {
		//
	} else {
		condNum = defaultCondNum * -1;
		//
	}
	*/

	var condNum = 0; // random(0, 1);

	var otherDecisionOrder = [[
	'<table style="display:inline-block"><tbody><tr>' +
	'<td><input id="Other_C2" type="radio" name="BTS_actual-otherPlayer-confidence" value="0" />' +
	'<label class="confidece-image C2" for="Other_C2"></label></td>' +
	'<td><input id="Other_C1" type="radio" name="BTS_actual-otherPlayer-confidence" value="1" />' +
	'<label class="confidece-image C1" for="Other_C1"></label></td>' +
	'<td><input id="Other_C0" type="radio" name="BTS_actual-otherPlayer-confidence" value="2" />' +
	'<label class="confidece-image C0" for="Other_C0"></label></td>' +
	'<td style="width: 30px">&nbsp;&nbsp;</td>' +
	'<td><input id="Other_D0" type="radio" name="BTS_actual-otherPlayer-confidence" value="3" />' +
	'<label class="confidece-image D0" for="Other_D0"></label></td>' +
	'<td><input id="Other_D1" type="radio" name="BTS_actual-otherPlayer-confidence" value="4" />' +
	'<label class="confidece-image D1" for="Other_D1"></label></td>' +
	'<td><input id="Other_D2" type="radio" name="BTS_actual-otherPlayer-confidence" value="5" />' +
	'<label class="confidece-image D2" for="Other_D2"></label></td>' +
	'</tr></tbody><table>'
	],[
	'<table style="display:inline-block"><tbody><tr>' +
	'<td><input id="Other_C2-predicted" type="radio" name="BTS_predicted-otherPlayer-confidence" value="0" />' +
	'<label class="confidece-image C2" for="Other_C2-predicted"></label></td>' +
	'<td><input id="Other_C1-predicted" type="radio" name="BTS_predicted-otherPlayer-confidence" value="1" />' +
	'<label class="confidece-image C1" for="Other_C1-predicted"></label></td>' +
	'<td><input id="Other_C0-predicted" type="radio" name="BTS_predicted-otherPlayer-confidence" value="2" />' +
	'<label class="confidece-image C0" for="Other_C0-predicted"></label></td>' +
	'<td style="width: 30px">&nbsp;&nbsp;</td>' +
	'<td><input id="Other_D0-predicted" type="radio" name="BTS_predicted-otherPlayer-confidence" value="3" />' +
	'<label class="confidece-image D0" for="Other_D0-predicted"></label></td>' +
	'<td><input id="Other_D1-predicted" type="radio" name="BTS_predicted-otherPlayer-confidence" value="4" />' +
	'<label class="confidece-image D1" for="Other_D1-predicted"></label></td>' +
	'<td><input id="Other_D2-predicted" type="radio" name="BTS_predicted-otherPlayer-confidence" value="5" />' +
	'<label class="confidece-image D2" for="Other_D2-predicted"></label></td>' +
	'</tr></tbody><table>'
	]];

	var thisDecisionOrder = [[
	'<table style="display:inline-block"><tbody><tr>' +
	'<td><input id="This_C2" type="radio" name="BTS_actual-thisPlayer-confidence" value="0" />' +
	'<label class="confidece-image C2" for="This_C2"></label></td>' +
	'<td><input id="This_C1" type="radio" name="BTS_actual-thisPlayer-confidence" value="1" />' +
	'<label class="confidece-image C1" for="This_C1"></label></td>' +
	'<td><input id="This_C0" type="radio" name="BTS_actual-thisPlayer-confidence" value="2" />' +
	'<label class="confidece-image C0" for="This_C0"></label></td>' +
	'<td style="width: 30px">&nbsp;&nbsp;</td>' +
	'<td><input id="This_D0" type="radio" name="BTS_actual-thisPlayer-confidence" value="3" />' +
	'<label class="confidece-image D0" for="This_D0"></label></td>' +
	'<td><input id="This_D1" type="radio" name="BTS_actual-thisPlayer-confidence" value="4" />' +
	'<label class="confidece-image D1" for="This_D1"></label></td>' +
	'<td><input id="This_D2" type="radio" name="BTS_actual-thisPlayer-confidence" value="5" />' +
	'<label class="confidece-image D2" for="This_D2"></label></td>' +
	'</tr></tbody><table>'
	],[
	'<table style="display:inline-block"><tbody><tr>' +
	'<td><input id="This_C2-predicted" type="radio" name="BTS_predicted-thisPlayer-confidence" value="0" />' +
	'<label class="confidece-image C2" for="This_C2-predicted"></label></td>' +
	'<td><input id="This_C1-predicted" type="radio" name="BTS_predicted-thisPlayer-confidence" value="1" />' +
	'<label class="confidece-image C1" for="This_C1-predicted"></label></td>' +
	'<td><input id="This_C0-predicted" type="radio" name="BTS_predicted-thisPlayer-confidence" value="2" />' +
	'<label class="confidece-image C0" for="This_C0-predicted"></label></td>' +
	'<td style="width: 30px">&nbsp;&nbsp;</td>' +
	'<td><input id="This_D0-predicted" type="radio" name="BTS_predicted-thisPlayer-confidence" value="3" />' +
	'<label class="confidece-image D0" for="This_D0-predicted"></label></td>' +
	'<td><input id="This_D1-predicted" type="radio" name="BTS_predicted-thisPlayer-confidence" value="4" />' +
	'<label class="confidece-image D1" for="This_D1-predicted"></label></td>' +
	'<td><input id="This_D2-predicted" type="radio" name="BTS_predicted-thisPlayer-confidence" value="5" />' +
	'<label class="confidece-image D2" for="This_D2-predicted"></label></td>' +
	'</tr></tbody><table>'
	]];

	var potSizeEst = [
	'<table style="display:inline-block"><tbody><tr>'+
	'<td><input id="Pot_small" type="radio" name="potSize-estimate" value="0" />' +
	'<label class="confidece-image pot0" for="Pot_small"></label></td>' +
	'<td><input id="Pot_medium" type="radio" name="potSize-estimate" value="1" />' +
	'<label class="confidece-image pot1" for="Pot_medium"></label></td>' +
	'<td><input id="Pot_large" type="radio" name="potSize-estimate" value="2" />' +
	'<label class="confidece-image pot2" for="Pot_large"></label></td>'+
	'</tr></tbody></table>'
	];

	var choiceBall1 = [
		'images/SplitBall.png',
		'images/StoleBall.png'
		//'<td style="display: block; margin: auto;"><img width="70" height="70" src="images/StoleBall.png"></td>'
	];
	var choiceBall2 = [
		'images/StoleBall.png',
		'images/SplitBall.png'
		//'<td style="display: block; margin: auto;"><img width="70" height="70" src="images/StoleBall.png"></td>'
	];

	$("#BTS_actual_otherPlayDecisionOptions").html(otherDecisionOrder[0][condNum]);
	// $("#BTS_predicted_otherPlayDecisionOptions").html(otherDecisionOrder[1][condNum]);
	$("#BTS_actual_thisPlayDecisionOptions").html(thisDecisionOrder[0][condNum]);
	// $("#BTS_predicted_thisPlayDecisionOptions").html(thisDecisionOrder[1][condNum]);
	// $("#potSizePlayOptions").html(potSizeEst[condNum]);

	document.getElementById("choiceBallMain-left-this").src = choiceBall1[condNum];
	document.getElementById("choiceBallMain-right-this").src = choiceBall2[condNum];
	document.getElementById("choiceBallMain-left-other").src = document.getElementById("choiceBallMain-left-this").src;
	document.getElementById("choiceBallMain-right-other").src = document.getElementById("choiceBallMain-right-this").src;

	maintaskParam = new SetMaintaskParam();

	// Updates the progress bar
	$("#trial-num").html(maintaskParam.numComplete);
	$("#total-num").html(maintaskParam.numTrials);


	maintask = {

		stimIDArray: new Array(maintaskParam.numTrials),
		stimulusArray: new Array(maintaskParam.numTrials),

		BTS_actual_otherDecisionConfidence: new Array(maintaskParam.numTrials),
		BTS_actual_thisDecisionConfidence: new Array(maintaskParam.numTrials),
		BTS_actual_payoffQuadrant: new Array(maintaskParam.numTrials),
		BTS_predicted_otherDecisionConfidence: new Array(maintaskParam.numTrials),
		BTS_predicted_thisDecisionConfidence: new Array(maintaskParam.numTrials),
		// BTS_predicted_payoffQuadrant: new Array(maintaskParam.numTrials),
		// potSizeConfidence: new Array(maintaskParam.numTrials),

		randCondNum: new Array(1),
		quadResult: new Array(1),
		// quadComplete: new Array(1),

		validationRadioExpectedResp: new Array(2),
		validationRadio: new Array(3),
		dem_gender: [],
		dem_language: [],
		val_recognized: [],
		val_feedback: [],
		val_email: [],

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
			var quads = [2,3,4,4,1,1,4,4,1,1,3,2,1,1,1,1,4,4,2,3,3,2,1,1,1,1,3,2,2,3,4,4,
			1,1,2,3,1,1,4,4,4,4,2,3,3,2,3,2,4,4,1,1,3,2,3,2,3,2,3,2,1,1,3,2,3,2,3,2,1,1,
			3,2,4,4,3,2,2,3,4,4,4,4,3,2,4,4,2,3,];
			var quadSums = new Array(quads.length);

			// if(maintask.dataInSitu[iQuad].hasOwnProperty('BTS_actual_payoffQuadrant'))
			// {
			//     // Do something
			// }
			if (maintask.dataInSitu.length == quads.length) {
				for (var iQuad = 0; iQuad < quads.length; iQuad++) {
					if (maintask.dataInSitu[iQuad].BTS_actual_payoffQuadrant == quads[iQuad]) { // "1" == 1 -> true; === -> false
						quadSums[iQuad] = 1;
					} else {quadSums[iQuad] = 0;}

					if (iQuad == quads.length - 1) {
						this.quadResult = sumArray(quadSums);
					}
				}	
			}

			this.dem_gender = $('input[name="d1"]:checked').val();
			this.dem_language = $('textarea[name="dem_language"]').val();
			this.val_recognized = $('textarea[name="val_recognized"]').val();
			this.val_feedback = $('textarea[name="val_feedback"]').val();
			this.val_email = $('textarea[name="val_email"]').val();

			// SEND DATA TO TURK
			setTimeout(function() {
				turk.submit(maintask, true, mmtofurkeyGravy);
				setTimeout(function() { showSlide("exit"); }, 1000);
			}, 1000);

			/*
			// DEBUG PUT THIS IN MMTOFURKYGRAVY AND ADD STRING PARAM TO MAINTASK VARIABLES
			console.log("attempting to return condition");
			var returnServe = 'nReps=' + phpParam.nReps + '&writestatus=' + phpParam.writestatus + '&condComplete=' + maintask.randCondNum.toString() + '&condfname=' + phpParam.condfname;
			$.getJSON(phpParam.baseURL, returnServe, function(res) {
				console.log("Serve Returned!", res.condNum);
			});
			*/

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
			$("#videoStimPackageDiv").hide();
			$("#playButtonContainer").show();
			showSlide("slideStimulus");
			responseDivCues1.style.display = 'block';

			// slideStimulusContext
			if (!!revoke) {
				smallVideo.src = URL.revokeObjectURL(smallVideo.src); // IE10+
				smallVideo.src = '';
				thisVideo.src = URL.revokeObjectURL(smallVideo.src); // IE10+
				thisVideo.src = '';
			}

			// duplicate allTrialOrders
			if (maintaskParam.numComplete === 0) {
				// this.randCondNum.push(condNum); // push randomization number
				this.randCondNum = condNum;
				disablePlayButton();
				if (!!debug) {
					preloadStim(0,false); // load first video direct
				} else {
					preloadStim(0,true); // load first video as blob
				}
				
				document.getElementById("imageStim_preload").src = serverRoot + stimPath + "statics/" + maintaskParam.allTrialOrders[maintaskParam.shuffledOrder[0]].stimulus + ".png"; // first static image

				if (!!maintaskParam.storeDataInSitu) {
					for (var i = 0; i < maintaskParam.allTrialOrders.length; i++) {
						var temp = maintaskParam.allTrialOrders[i];
						// temp.q1responseArray = "";

						this.dataInSitu.push(temp);
						//// CLEAN this up a little?
					}
				}
			}

			// If this is not the first trial, record variables
			if (maintaskParam.numComplete > 0) {

				// Adds current answers to arrays of that answer type (after answers are collected)
				this.BTS_actual_otherDecisionConfidence[maintaskParam.numComplete - 1] = getRadioResponse('BTS_actual-otherPlayer-confidence');
				this.BTS_actual_thisDecisionConfidence[maintaskParam.numComplete - 1] = getRadioResponse('BTS_actual-thisPlayer-confidence');
				this.BTS_actual_payoffQuadrant[maintaskParam.numComplete - 1] = updatePayoffMatrix(0);
				this.BTS_predicted_otherDecisionConfidence[maintaskParam.numComplete - 1] = document.getElementById("confidenceOther").value;
				this.BTS_predicted_thisDecisionConfidence[maintaskParam.numComplete - 1] = document.getElementById("confidenceThis").value;

				// Adds answers to .data and .dataInSitu
				maintaskParam.trial.BTS_actual_otherDecisionConfidence = getRadioResponse('BTS_actual-otherPlayer-confidence');
				maintaskParam.trial.BTS_actual_thisDecisionConfidence = getRadioResponse('BTS_actual-thisPlayer-confidence');
				maintaskParam.trial.BTS_actual_payoffQuadrant = updatePayoffMatrix(0);
				// maintaskParam.trial.BTS_predicted_otherDecisionConfidence = getRadioResponse('BTS_predicted-otherPlayer-confidence');
				// maintaskParam.trial.BTS_predicted_thisDecisionConfidence = getRadioResponse('BTS_predicted-thisPlayer-confidence');
				maintaskParam.trial.BTS_predicted_otherDecisionConfidence = document.getElementById("confidenceOther").value;
				maintaskParam.trial.BTS_predicted_thisDecisionConfidence = document.getElementById("confidenceThis").value;
				// maintaskParam.trial.BTS_predicted_payoffQuadrant = updatePayoffMatrix(1);

				// maintaskParam.trial.potSizeConfidence = getRadioResponse('potSize-estimate');

				this.data.push(maintaskParam.trial);

				//// Reset forum elements
				ResetRanges();

				ResetRadios('BTS_actual-otherPlayer-confidence');
				ResetRadios('BTS_actual-thisPlayer-confidence');
				ResetRadios('BTS_predicted-otherPlayer-confidence');
				ResetRadios('BTS_predicted-thisPlayer-confidence');

				// ResetRadios('potSize-estimate');

				// Reset display elements
				document.getElementById("choiceBallMain-left-other").style.opacity = 1;
				document.getElementById("choiceBallMain-right-other").style.opacity = 1;
				document.getElementById("choiceBallMain-left-this").style.opacity = 1;
				document.getElementById("choiceBallMain-right-this").style.opacity = 1;

				// Reset checkbox toggle
				document.getElementById("BTS_toggle").checked = false;
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

				/// document.getElementById("videoStim").src = serverRoot + stimPath + "dynamics/" + maintaskParam.allTrialOrders[maintaskParam.shuffledOrder[maintaskParam.numComplete]].stimulus + "t.mp4";
				/// enablePlayButton();

				// document.getElementById("imageStim").src = serverRoot + stimPath + "statics/" + maintaskParam.trial.stimulus + ".png"; // preload next static image
				document.getElementById("imageStim").src = document.getElementById("imageStim_preload").src;


				$("#responseDivCues1").html('<p><b>Jackpot:&nbsp;&nbsp;&nbsp;$&nbsp;' + numberWithCommas(maintaskParam.trial.pot) + '</b></p>');
				$("#moneytotal").html('Jackpot:<br>$' + numberWithCommas(maintaskParam.trial.pot));

				// if (maintaskParam.trial.stimulus[4] == 1) {
				// 	document.getElementById("imageStim_front1").src = document.getElementById("imageStim").src;
				// 	document.getElementById("imageStim_front2").src = serverRoot + "images/generic_avatar_male.png";
				// 	maintaskParam.trial.decisionOther
				// } else if (maintaskParam.trial.stimulus[4] == 2) {
				// 	document.getElementById("imageStim_front1").src = serverRoot + "images/generic_avatar_male.png";
				// 	document.getElementById("imageStim_front2").src = document.getElementById("imageStim").src;
				// }


				/// console.log("currentImg", maintaskParam.trial.stimulus);
				// $('#context_jackpot').text("$" + numberWithCommas(maintaskParam.trial.pot));
				// $('#context_jackpot_front').text("$" + numberWithCommas(maintaskParam.trial.pot));

				// $('#contextText_decisionOther').html("&nbsp;" + maintaskParam.trial.decisionOther);
				// $('#contextText_decisionThis').html("&nbsp;" + maintaskParam.trial.decisionThis);
				// document.getElementById("contextImg_decisionOther").src = serverRoot + "images/" + maintaskParam.trial.decisionOther + "Ball.png";
				// document.getElementById("contextImg_decisionThis").src = serverRoot + "images/" + maintaskParam.trial.decisionThis + "Ball.png";
				document.getElementById("miniface_Other").src = serverRoot + "images/generic_avatar_male.png";
				document.getElementById("miniface_This").src = document.getElementById("imageStim_preload").src;

				document.getElementById("minifacePM_Other").src = document.getElementById("miniface_Other").src;
				document.getElementById("minifacePM_This").src = document.getElementById("miniface_This").src;

				/*
				var outcomeOther = 0;
				var outcomeThis = 0;
				if (maintaskParam.trial.decisionOther === "Split" && maintaskParam.trial.decisionThis === "Split") {
					outcomeOther = 'Won $' + numberWithCommas(Math.floor(maintaskParam.trial.pot * 50) / 100);
					outcomeThis = outcomeOther;

					document.getElementById("imageContext").src = serverRoot + "images/" + "CC.png";
					$('#context_outcome_front1').html(outcomeOther);
					$('#context_outcome_front2').html(outcomeOther);

				}
				if (maintaskParam.trial.decisionOther === "Split" && maintaskParam.trial.decisionThis === "Stole") {
					outcomeOther = 'Won $0.00';
					outcomeThis = 'Won $' + numberWithCommas(maintaskParam.trial.pot);

					if (maintaskParam.trial.stimulus[4] == 1) {
						document.getElementById("imageContext").src = serverRoot + "images/" + "DC.png";
						$('#context_outcome_front1').html(outcomeThis);
						$('#context_outcome_front2').html(outcomeOther);
					} else if (maintaskParam.trial.stimulus[4] == 2) {
						document.getElementById("imageContext").src = serverRoot + "images/" + "CD.png";
						$('#context_outcome_front1').html(outcomeOther);
						$('#context_outcome_front2').html(outcomeThis);
					}


				}
				if (maintaskParam.trial.decisionOther === "Stole" && maintaskParam.trial.decisionThis === "Split") {
					outcomeOther = 'Won $' + numberWithCommas(maintaskParam.trial.pot);
					outcomeThis = 'Won $0.00';

					if (maintaskParam.trial.stimulus[4] == 1) {
						document.getElementById("imageContext").src = serverRoot + "images/" + "CD.png";
						$('#context_outcome_front1').html(outcomeThis);
						$('#context_outcome_front2').html(outcomeOther);
					} else if (maintaskParam.trial.stimulus[4] == 2) {
						document.getElementById("imageContext").src = serverRoot + "images/" + "DC.png";
						$('#context_outcome_front1').html(outcomeOther);
						$('#context_outcome_front2').html(outcomeThis);
					}

				}
				if (maintaskParam.trial.decisionOther === "Stole" && maintaskParam.trial.decisionThis === "Stole") {
					outcomeOther = 'Won $0.00';
					outcomeThis = 'Won $0.00';

					document.getElementById("imageContext").src = serverRoot + "images/" + "DD.png";
					$('#context_outcome_front1').html(outcomeOther);
					$('#context_outcome_front2').html(outcomeOther);
				}
				$('#context_outcomeOther').html(outcomeOther);
				$('#context_outcomeThis').html(outcomeThis);

				$("#contextTableFrontDiv").html('&nbsp;');
				$("#contextSubTableID").clone().appendTo("#contextTableFrontDiv"); // insert information in video div
				*/

				// Adds current answers to arrays of that answer type (before answers are collected)
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

	// });
}
