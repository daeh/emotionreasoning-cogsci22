
// It is handy to not have those transformations while we developing
if(process.env.NODE_ENV === 'production') {
	module.exports = {
		plugins: [
			require('autoprefixer'),
			require('cssnano'),
			// More postCSS modules here if needed
		]
	}
} else if(process.env.NODE_ENV === 'development') {
	module.exports = {
		plugins: [
			require('autoprefixer'),
			// More postCSS modules here if needed
		]
	}
}