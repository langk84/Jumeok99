var express = require('express')
  , http = require('http')
  , path = require('path');
var bodyParser = require('body-parser')
  , static = require('serve-static');
var app = express();
app.set('port', process.env.PORT || 80);

app.use(bodyParser.urlencoded({ extended: false }))

app.use(bodyParser.json())

app.use(static(path.join(__dirname, 'public')));
app.use(function(req, res, next) {
	console.log('요청받음');

	var paramId = req.body.id || req.query.id;
	var paramPassword = req.body.password || req.query.password;
	
	res.writeHead('200', {'Content-Type':'text/html;charset=utf8'});
	res.write('<h1>요청 수락됨</h1>');
	res.write('<div><p>요청한 주식 : ' + paramId + '</p></div>');
	res.write('<div><p>학습 횟수 : ' + paramPassword + '</p></div>');
    res.write('<div><p>결과 페이지localhost/'+paramId+','+paramPassword+'.png</p></div>');
	res.end();
    console.log(paramId, paramPassword);
    const fs = require('fs');
    fs.writeFile('require.txt', paramId+'\n'+paramPassword, (err) => {
        if (err) console.log('Error: ', err);
        else console.log('File created');
});
    fs.writeFile('check.i', ' ', (err) => {
    if (err) console.log('Error: ', err);
    else console.log('check');
});
});


// Express 서버 시작
http.createServer(app).listen(80, function(){
  console.log('start server');
});