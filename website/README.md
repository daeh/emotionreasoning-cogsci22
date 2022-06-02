



### Webpack init

```sh
cd "${HOME}/coding/-GitRepos/itegb_cuecomb_cogsci2022/website" || exit
# npm install ### install packages in `package.json`
npm install core-js jquery
npm install --save-dev @babel/core @babel/preset-env babel-loader copy-webpack-plugin css-loader file-loader html-webpack-plugin less-loader mini-css-extract-plugin postcss-loader postcss-preset-env sass sass-loader webpack webpack-cli url-loader
```

## Add to `package.json`

```json
  "name": "emotionreasoning-cogsci22",
  "version": "0.0.1",
  "description": "",
  "main": "index.js",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1",
    "build": "webpack --mode=development",
    "produce": "webpack --mode=production"
  },
  "keywords": [],
  "author": "Dae Houlihan",
  "license": "ISC",
  "browserslist": [
    "> 5% in US",
    "last 2 versions"
  ],
```



## Push changes

```zsh
#wpspec="produce"
#wpspec="build"
cd "${HOME}/coding/-GitRepos/itegb_cuecomb_cogsci2022/website" || exit
npm run "${wpspec}"

rm -r "${HOME}/coding/-GitRepos/daehinfo_website/jekyll-TeXt-theme-master/p/emotionreasoning-cogsci2022"

cp -r "${HOME}/coding/-GitRepos/itegb_cuecomb_cogsci2022/website/dist" "${HOME}/coding/-GitRepos/daehinfo_website/jekyll-TeXt-theme-master/p/emotionreasoning-cogsci2022"
```

