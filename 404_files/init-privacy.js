jQuery( document ).ready(function( $ ) {
  window.$ = $;
  s = new Privacy({
        btn_text: "Accetta!",
        link: "http://www.plasm.it/cookie-policy",
        position: "top",
        opt_cookie: {
            domain: "plasm.it"
        },
        ga: {
            id: "UA-15267220-3",
            domain: "auto"
        },
        onClose: function(){
            $(window).trigger("resize")
        }
    })
});
