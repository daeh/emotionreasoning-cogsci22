const path = require('path');
const MiniCssExtractPlugin = require("mini-css-extract-plugin");
const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyWebpackPlugin = require('copy-webpack-plugin');
// const isModern = process.env.BROWSERSLIST_ENV === 'modern';

/// specify browser support
/// useful https://dev.to/pixelgoo/how-to-configure-webpack-from-scratch-for-a-basic-website-46a5
/// babel browser support https://www.thebasement.be/working-with-babel-7-and-webpack/
/// The cool part about the browserslist is that you define your browsers in one place, and other tools like postCss will use that same resource. So your supported browsers are defined in a single source of truth, which is a best practice
/// https://sgom.es/posts/2019-03-06-supporting-old-browsers-without-hurting-everyone/
/// https://www.smashingmagazine.com/2018/10/smart-bundling-legacy-code-browsers/


/// typical set of config files
///   postcss.config.js
///   babel.config.js
///   .eslintrc.js
///   .editorconfig
///   webpack.config.js

module.exports = {
  entry: './src/experiment.js',
  mode: 'development',
  output: {
    filename: 'main.js',
    path: path.resolve(__dirname, 'dist'),
  },

  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /(node_modules)/,
        use: {
          loader: 'babel-loader',
          // options: { presets: ['@babel/preset-env'] } /// specified in .babelrc /// Important: The docs are a bit unclear on this one, but if you set your @babel/preset-env-options in the .babelrc-file you don’t have to define them in your Webpack configuration. If you do so, the Webpack configuration will overwrite the options in your .babelrc
        }
      },
      {
        // Apply rule for .sass, .scss or .css files
        test: /\.(sa|sc|c)ss$/,

        // Set loaders to transform files.
        // Loaders are applying from right to left(!)
        // The first loader will be applied after others
        use: [
          {
            // The purpose of plugins like MiniCssExtractPlugin is to do anything else that loaders can't. If we need to extract all that transformed CSS into a separate "bundle" file we have to use a plugin. And there is a special one for our case: MiniCssExtractPlugin:
            // After all CSS loaders we use plugin to do his work.
            // It gets all transformed CSS and extracts it into separate
            // single bundled file
            loader: MiniCssExtractPlugin.loader,
          },
          {
            // This loader resolves url() and @imports inside CSS
            loader: "css-loader",
          },
          {
            // Then we apply postCSS fixes like autoprefixer and minifying
            loader: "postcss-loader",
          },
          {
            // First we transform SASS to standard CSS
            loader: "sass-loader",
            options: {
              implementation: require("sass")
            }
          }
        ]
      },
      {
        // Apply rule for .less files
        test: /\.less$/,

        // Set loaders to transform files.
        // Loaders are applying from right to left(!)
        // The first loader will be applied after others
        use: [
          {
            // The purpose of plugins like MiniCssExtractPlugin is to do anything else that loaders can't. If we need to extract all that transformed CSS into a separate "bundle" file we have to use a plugin. And there is a special one for our case: MiniCssExtractPlugin:
            // After all CSS loaders we use plugin to do his work.
            // It gets all transformed CSS and extracts it into separate
            // single bundled file
            loader: MiniCssExtractPlugin.loader,
          },
          {
            // This loader resolves url() and @imports inside CSS
            loader: "css-loader",
          },
          {
            // Then we apply postCSS fixes like autoprefixer and minifying
            loader: "postcss-loader",
          },
          {
            // First we transform LESS to standard CSS
            loader: "less-loader", // compiles Less to CSS
          }
        ]
      },
      {
        test: /\.(jpg|jpeg|png|gif|svg|bmp)$/,
        use: [
          // { loader: 'url-loader?limit=100000', },
          {
          loader: "file-loader",
          options: {
            name: "[name].[ext]",
            outputPath: "assets/images"
          }
        }
        ]
      }
    ]
  },
  plugins: [
  new MiniCssExtractPlugin({filename: "[name].css"}),
  new HtmlWebpackPlugin({
      inject: false,
      hash: true,
      template: './src/index.html',
      filename: 'index.html'
    }),
  ] // "[name]-[contenthash:8].css"
};