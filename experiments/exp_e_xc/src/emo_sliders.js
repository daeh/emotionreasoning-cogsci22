
/// initialize sliders ///
document.querySelectorAll('.not-clicked').forEach(slider => slider.addEventListener('click', e => {
	let label_naive = e.currentTarget.closest('.slider').querySelector('.slider-label-not-clicked');
	e.currentTarget.classList.replace('not-clicked', 'slider-input');
	if (label_naive !== null) {
		label_naive.classList.replace('slider-label-not-clicked', 'slider-label');
	}
}));

document.querySelectorAll('.emotionRatingRow').forEach(erow => {

	erow.onmouseover = erow.onmouseenter = erow.onmouseout = erow.onmouseleave = (event) => {

		let floorcell = event.currentTarget.querySelector('.eFloor');
		let ceilcell = event.currentTarget.querySelector('.eCeiling');

		if (event.type === 'mouseover' || event.type === 'mouseenter') {

			/// initialize rollover effects of emotion labels ///

			floorcell.textContent = 'not any';
			floorcell.addEventListener('click', ee => {
				let thisSlider = ee.currentTarget.closest('tr').querySelector('input');
				thisSlider.value = "0";
				if (thisSlider.classList.contains('not-clicked')) {
					thisSlider.classList.replace('not-clicked', 'slider-input');
				}
				let thisSliderLabel = ee.currentTarget.closest('tr').querySelector('label');
				if (thisSliderLabel.classList.contains('slider-label-not-clicked')) {
					thisSliderLabel.classList.replace('slider-label-not-clicked', 'slider-label');
				}

			});

			ceilcell.textContent = 'immense';
			ceilcell.addEventListener('click', ee => {
				let thisSlider = ee.currentTarget.closest('tr').querySelector('input');
				thisSlider.value = "48";
				if (thisSlider.classList.contains('not-clicked')) {
					thisSlider.classList.replace('not-clicked', 'slider-input');
				}
				let thisSliderLabel = ee.currentTarget.closest('tr').querySelector('label');
				if (thisSliderLabel.classList.contains('slider-label-not-clicked')) {
					thisSliderLabel.classList.replace('slider-label-not-clicked', 'slider-label');
				}

			})

		}
		if (event.type === 'mouseout' || event.type === 'mouseleave') {
			floorcell.textContent = ' ';
			ceilcell.textContent = ' ';
		}

	}
});