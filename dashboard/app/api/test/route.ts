import { NextRequest, NextResponse } from 'next/server'
import { getAllUsers, searchUsers, getSystemStats } from '@/lib/dataService'

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const action = searchParams.get('action') || 'stats'
    
    switch (action) {
      case 'stats':
        const stats = await getSystemStats()
        return NextResponse.json({
          success: true,
          data: stats
        })
        
      case 'drivers':
        const drivers = await getAllUsers('Driver')
        return NextResponse.json({
          success: true,
          data: drivers.slice(0, 10), // First 10 for testing
          total: drivers.length
        })
        
      case 'merchants':
        const merchants = await getAllUsers('Merchant')
        return NextResponse.json({
          success: true,
          data: merchants.slice(0, 10), // First 10 for testing
          total: merchants.length
        })
        
      case 'search':
        const term = searchParams.get('term') || ''
        const type = searchParams.get('type') as 'Driver' | 'Merchant' || 'Driver'
        const results = await searchUsers(term, type)
        return NextResponse.json({
          success: true,
          data: results,
          count: results.length,
          searchTerm: term,
          userType: type
        })
        
      default:
        return NextResponse.json({
          success: false,
          error: 'Invalid action parameter'
        }, { status: 400 })
    }
  } catch (error) {
    console.error('Test API Error:', error)
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred',
      stack: error instanceof Error ? error.stack : undefined
    }, { status: 500 })
  }
}
