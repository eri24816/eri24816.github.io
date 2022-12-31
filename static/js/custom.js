onscroll = function () {
    scrollTop = $(window).scrollTop();
    opacity = scrollTop / 300;
    blur = Math.min(2, scrollTop / 400);
    // if .homepage exists, then we are on the homepage
    if ($('.homepage').length) {
        brightness = Math.max(0.2, 0.7 - scrollTop / 300);
    }
    else {
        brightness = 0.2;
    }

    ['.articles'].forEach( function(e){
        $(e).css('opacity', opacity);
    })
    
    //$('body').css('backdrop-filter', `brightness(${brightness}) blur(${blur}px)`);
    $('body').css('backdrop-filter', `brightness(${brightness})`);
    $('body').css('background-position', `center ${-0.3 * scrollTop}px`);
}

$(window).scroll(onscroll);

$(document).ready(onscroll); 