var mousePressed = false;
var lastX, lastY;
var ctx;

function InitThis() {
	  ctx= document.getElementById('myCanvas').getContext("2d");
		var w = ctx.canvas.width;
    var h = ctx.canvas.height;
		ctx.fillStyle = '#ffffff';
		ctx.fillRect(0,0,w,h);

    $('#myCanvas').mousedown(function (e) {
        mousePressed = true;
        Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
    });

    $('#myCanvas').mousemove(function (e) {
        if (mousePressed) {
            Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
        }
    });

    $('#myCanvas').mouseup(function (e) {
        mousePressed = false;
    });
    $('#myCanvas').mouseleave(function (e) {
        mousePressed = false;
    });
}

function Draw(x, y, isDown) {
    if (isDown) {
        ctx.beginPath();
        ctx.strokeStyle = $('#selColor').val();
        ctx.lineWidth = $('#selWidth').val();
        ctx.lineJoin = "round";
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.closePath();
        ctx.stroke();
    }
    lastX = x; lastY = y;
}

function clearArea() {
    // Use the identity matrix while clearing the canvas
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
		var w = ctx.canvas.width;
		var h = ctx.canvas.height;
		ctx.fillStyle = '#ffffff';
		ctx.fillRect(0,0,w,h);
}

function UploadPic(){
	var Pic = document.getElementById("myCanvas").toDataURL("image/png");
    Pic = Pic.replace(/^data:image\/(png|jpg);base64,/, "")
	$.ajax({
		type:'POST',
		url:'',
		data:'{"image":"'+Pic+'"}',
		contentType:'application/json;charset=utf-8',
		dataType:'json',
		success:function(msg){
			$('#recgResult').val(msg);
		}
	});

}

window.onload = function (){
	InitThis();
}
