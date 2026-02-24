! coul90.f90 - Coulomb functions wrapper for f2py
! Based on A.R. Barnett's COUL90

module coul90_mod
  implicit none
  private
  public :: coul90, coulph

contains

  subroutine coulph(eta, cph, lmaxc)
    ! Calculate Coulomb phase shifts sigma_l(eta)
    implicit none
    real*8, intent(in) :: eta
    integer, intent(in) :: lmaxc
    real*8, intent(out) :: cph(0:lmaxc)
    real*8 :: sto, fi
    integer :: ii

    sto = 16.d0 + eta*eta
    cph(0) = -eta + (eta/2.d0)*dlog(sto) + 3.5d0*datan(eta/4.d0) - &
             (datan(eta) + datan(eta/2.d0) + datan(eta/3.d0)) - &
             (eta/(12.d0*sto))*(1.d0 + (1.d0/30.d0)*(eta**2-48.d0)/sto**2 + &
             (1.d0/105.d0)*(eta**4-160.d0*eta**2+1280.d0)/sto**4)

    do ii = 1, lmaxc
       fi = dble(ii)
       cph(ii) = cph(ii-1) + datan(eta/fi)
    end do
  end subroutine coulph


  subroutine coul90(x, eta, xlmin, lrange, fc, gc, fcp, gcp, kfn, ifail)
    ! Coulomb & Bessel function program using Steed's method
    ! Author: A.R. Barnett
    implicit none
    integer, intent(in) :: lrange, kfn
    real*8, intent(in) :: x, eta, xlmin
    real*8, intent(out) :: fc(0:lrange), gc(0:lrange), fcp(0:lrange), gcp(0:lrange)
    integer, intent(out) :: ifail

    ! Local variables
    real*8 :: accur, acch, small, one, zero, half, two, ten2, rt2dpi
    real*8 :: xinv, pk, cf1, c, d, pk1, etak, rk2, tk, dcf1, den, xlm, xll
    real*8 :: el, xl, rl, sl, f, fcmaxl, fcminl, gcminl, omega, wronsk
    real*8 :: wi, a, b, ar, ai, br, bi, dr, di, dp, dq, alpha, beta
    real*8 :: e2mm1, fjwkb, gjwkb, p, q, paccq, gamma, gammai
    integer :: iexp, nfp, npq, l, minl, maxl, limit
    logical :: etane0, xlturn
    real*8 :: eta_local

    parameter(limit = 20000, small = 1.0d-150)
    data zero, one, two, ten2, half /0.0d0, 1.0d0, 2.0d0, 1.0d2, 0.5d0/
    data rt2dpi /0.7978845608028654d0/

    accur = 1.0d-14
    ifail = 0
    iexp = 1
    npq = 0
    gjwkb = zero
    paccq = one

    eta_local = eta
    if (kfn .ne. 0) eta_local = zero
    etane0 = eta_local .ne. zero
    acch = dsqrt(accur)

    if (x .le. acch) then
       ifail = -1
       return
    endif

    if (kfn .eq. 2) then
       xlm = xlmin - half
    else
       xlm = xlmin
    endif

    if (xlm .le. -one .or. lrange .lt. 0) then
       ifail = -2
       return
    endif

    e2mm1 = xlm * xlm + xlm
    xlturn = x * (x - two * eta_local) .lt. e2mm1
    e2mm1 = e2mm1 + eta_local * eta_local
    xll = xlm + dfloat(lrange)

    minl = max0(idint(xlmin + accur), 0)
    maxl = minl + lrange

    ! Evaluate CF1 = f = dF/dx / F
    xinv = one / x
    den = one
    pk = xll + one
    cf1 = eta_local / pk + pk * xinv
    if (dabs(cf1) .lt. small) cf1 = small
    rk2 = one
    d = zero
    c = cf1

    do l = 1, limit
       pk1 = pk + one
       if (etane0) then
          etak = eta_local / pk
          rk2 = one + etak * etak
          tk = (pk + pk1) * (xinv + etak / pk1)
       else
          tk = (pk + pk1) * xinv
       endif
       d = tk - rk2 * d
       c = tk - rk2 / c
       if (dabs(c) .lt. small) c = small
       if (dabs(d) .lt. small) d = small
       d = one / d
       dcf1 = d * c
       cf1 = cf1 * dcf1
       if (d .lt. zero) den = -den
       pk = pk1
       if (dabs(dcf1 - one) .lt. accur) exit
    end do

    if (l .ge. limit) then
       ifail = 1
       return
    endif

    nfp = int(pk - xll - 1)
    f = cf1

    ! Downward recurrence
    if (lrange .gt. 0) then
       fcmaxl = small * den
       fcp(maxl) = fcmaxl * cf1
       fc(maxl) = fcmaxl
       xl = xll
       rl = one

       do l = maxl, minl+1, -1
          if (etane0) then
             el = eta_local / xl
             rl = dsqrt(one + el * el)
             sl = xl * xinv + el
             gc(l) = rl
             gcp(l) = sl
          else
             sl = xl * xinv
          endif
          fc(l-1) = (fc(l) * sl + fcp(l)) / rl
          fcp(l-1) = fc(l-1) * sl - fc(l) * rl
          xl = xl - one
       end do

       if (dabs(fc(minl)) .lt. accur*small) fc(minl) = accur * small
       f = fcp(minl) / fc(minl)
       den = fc(minl)
    endif

    ! Evaluate CF2 = p + i.q
    if (xlturn) call jwkb(x, eta_local, dmax1(xlm, zero), fjwkb, gjwkb, iexp)

    if (iexp .gt. 1 .or. gjwkb .gt. (one / (acch*ten2))) then
       omega = fjwkb
       gamma = gjwkb * omega
       p = f
       q = one
    else
       xlturn = .false.
       pk = zero
       wi = eta_local + eta_local
       p = zero
       q = one - eta_local * xinv
       ar = -e2mm1
       ai = eta_local
       br = two * (x - eta_local)
       bi = two
       dr = br / (br * br + bi * bi)
       di = -bi / (br * br + bi * bi)
       dp = -xinv * (ar * di + ai * dr)
       dq = xinv * (ar * dr - ai * di)

       do l = 1, limit
          p = p + dp
          q = q + dq
          pk = pk + two
          ar = ar + pk
          ai = ai + wi
          bi = bi + two
          d = ar * dr - ai * di + br
          di = ai * dr + ar * di + bi
          c = one / (d * d + di * di)
          dr = c * d
          di = -c * di
          a = br * dr - bi * di - one
          b = bi * dr + br * di
          c = dp * a - dq * b
          dq = dp * b + dq * a
          dp = c
          if (dabs(dp) + dabs(dq) .lt. (dabs(p) + dabs(q)) * accur) exit
       end do

       if (l .ge. limit) then
          ifail = 2
          return
       endif

       npq = int(pk / two)
       paccq = half * accur / dmin1(dabs(q), one)
       if (dabs(p) .gt. dabs(q)) paccq = paccq * dabs(p)

       gamma = (f - p) / q
       gammai = one / gamma
       if (dabs(gamma) .le. one) then
          omega = dsqrt(one + gamma * gamma)
       else
          omega = dsqrt(one + gammai * gammai) * dabs(gamma)
       endif
       omega = one / (omega * dsqrt(q))
       wronsk = omega
    endif

    ! Renormalize for Bessel functions
    if (kfn .eq. 1) then
       alpha = xinv
       beta = xinv
    elseif (kfn .eq. 2) then
       alpha = half * xinv
       beta = dsqrt(xinv) * rt2dpi
    else
       alpha = zero
       beta = one
    endif

    fcminl = dsign(omega, den) * beta
    if (xlturn) then
       gcminl = gjwkb * beta
    else
       gcminl = fcminl * gamma
    endif
    if (kfn .ne. 0) gcminl = -gcminl

    fc(minl) = fcminl
    gc(minl) = gcminl
    gcp(minl) = gcminl * (p - q * gammai - alpha)
    fcp(minl) = fcminl * (f - alpha)

    if (lrange .eq. 0) return

    ! Upward recurrence
    omega = beta * omega / dabs(den)
    xl = xlm
    rl = one

    do l = minl+1, maxl
       xl = xl + one
       if (etane0) then
          rl = gc(l)
          sl = gcp(l)
       else
          sl = xl * xinv
       endif
       gc(l) = ((sl - alpha) * gc(l-1) - gcp(l-1)) / rl
       gcp(l) = rl * gc(l-1) - (sl + alpha) * gc(l)
       fcp(l) = omega * (fcp(l) - alpha * fc(l))
       fc(l) = omega * fc(l)
    end do

  end subroutine coul90


  subroutine jwkb(x, eta, xl, fjwkb, gjwkb, iexp)
    ! JWKB approximation for Coulomb functions
    implicit none
    real*8, intent(in) :: x, eta, xl
    real*8, intent(out) :: fjwkb, gjwkb
    integer, intent(out) :: iexp

    real*8 :: zero, half, one, six, ten, rl35, aloge
    real*8 :: gh2, xll1, hll, hl, sl, rl2, gh, phi, phi10
    integer :: maxexp

    parameter(maxexp = 300)
    data zero, half, one, six, ten /0.0d0, 0.5d0, 1.0d0, 6.0d0, 10.0d0/
    data rl35, aloge /35.0d0, 0.4342945d0/

    gh2 = x * (eta + eta - x)
    xll1 = dmax1(xl * xl + xl, 0.0d0)
    if (gh2 + xll1 .le. zero) return

    hll = xll1 + six / rl35
    hl = dsqrt(hll)
    sl = eta / hl + hl / x
    rl2 = one + eta * eta / hll
    gh = dsqrt(gh2 + hll) / x
    phi = x*gh - half*(hl*dlog((gh + sl)**2 / rl2) - dlog(gh))
    if (eta .ne. zero) phi = phi - eta * datan2(x*gh, x - eta)
    phi10 = -phi * aloge
    iexp = int(phi10)

    if (iexp .gt. maxexp) then
       gjwkb = 10.0d0**(phi10 - dfloat(iexp))
    else
       gjwkb = dexp(-phi)
       iexp = 0
    endif
    fjwkb = half / (gh * gjwkb)

  end subroutine jwkb

end module coul90_mod
