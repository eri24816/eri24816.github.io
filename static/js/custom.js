onscroll = function () {
    //your code here
    scrollTop = $(window).scrollTop();
    opacity = scrollTop / 300;
    blur = Math.min(2, scrollTop / 400);
    brightness = Math.max(0.2, 1 - scrollTop / 300);

    ['.articles'].forEach( function(e){
        $(e).css('opacity', opacity);
    })

    //$('body').css('backdrop-filter', `brightness(${brightness}) blur(${blur}px)`);
    $('body').css('backdrop-filter', `brightness(${brightness})`);
    $('body').css('background-position', `center ${-0.15 * scrollTop}px`);
}

$(window).scroll(onscroll);

$(document).ready(onscroll); 